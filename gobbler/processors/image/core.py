import json
import os
from typing import Literal, Optional

import cv2
from openai import AzureOpenAI

import gobbler.constants as c
import gobbler.cred as cred
import gobbler.globals as glb
from gobbler.logger import logger
from gobbler.models.core import run_clip
from gobbler.models.utils import ClipScene
from gobbler.processors.image.models import Image
from gobbler.processors.image.utils import (
    sys_msg_desc_text,
    sys_msg_dsc_diagram,
    sys_msg_dsc_entities,
)
from gobbler.processors.interfaces import BaseProcessor
from gobbler.utils import (
    dump_usage_data,
    get_file_metadata,
    get_usage_file,
    make_pil_images,
    stringify_image,
)


class ImageProcessor(BaseProcessor):
    def __init__(self, scene_to_desc: Optional[dict[ClipScene, str]] = None):
        """
        Parameters:
        - scene_to_desc: User-provided mapping of scene types to descriptions,
            OCR API will not be used for these scene types.
        """
        if not cred.AZURE_VLM_KEY:
            raise EnvironmentError("Missing Azure VLM key")

        self.vlm_client = AzureOpenAI(
            api_key=cred.AZURE_VLM_KEY,
            azure_endpoint=cred.AZURE_VLM_BASE,
            azure_deployment=cred.AZURE_VLM_DEPLOYMENT,
            api_version=cred.AZURE_VLM_VERSION,
        )
        self.scene_to_desc = scene_to_desc or {}
        self.usage_file = get_usage_file(c.USAGE_AOAI_OCR)
        self.usage_data = {
            cred.AZURE_VLM_MODEL: {
                c.FLD_USAGE_PROMPT: 0,
                c.FLD_USAGE_COMPLETION: 0,
            }
        }

    def classify(
        self, path: str, heur_thresh: float = 1.5
    ) -> Optional[ClipScene]:
        scene_probs = run_clip(make_pil_images([path]))[0]
        top_scene, top_prob = scene_probs[0]
        if top_prob >= glb.clip_prob_thresh:
            return top_scene
        scene_probs = {scene: prob for scene, prob in scene_probs}
        text_prob = scene_probs[ClipScene.TEXT]
        diagram_prob = scene_probs[ClipScene.DIAGRAM]
        tab_prob = scene_probs[ClipScene.TABULAR]

        # heuristics
        # "text >> diagram >> tabular >> ..."" means text
        if (
            top_scene == ClipScene.TEXT
            and text_prob > (diagram_prob * heur_thresh)
            and diagram_prob > (tab_prob * heur_thresh)
            and (text_prob + diagram_prob + tab_prob) > glb.clip_prob_thresh
        ):
            return ClipScene.TEXT

        # "table >> text >> ..." or "text >> table >> ..." means table with text
        if (
            top_scene in (ClipScene.TABULAR, ClipScene.TEXT)
            and (
                tab_prob > (text_prob * heur_thresh)
                or text_prob > (tab_prob * heur_thresh)
            )
            and (tab_prob + heur_thresh) > glb.clip_prob_thresh
        ):
            return ClipScene.TABULAR

        return None

    def call_4o(
        self,
        sys_msg: str,
        img_content: str,
        fidelity: str = "auto",
        response_format: str = "json_object",
        temperature: float = 0.5,
    ) -> str:
        messages = [
            {
                c.LLM_FLD_ROLE: c.LLM_ROLE_SYSTEM,
                c.LLM_FLD_CONTENT: sys_msg,
            },
            {
                c.LLM_FLD_ROLE: c.LLM_ROLE_USER,
                c.LLM_FLD_CONTENT: [
                    {
                        c.LLM_FLD_TYPE: c.LLM_CONTENT_TYPE_IMAGE_URL,
                        c.LLM_FLD_IMAGE_URL: {
                            c.LLM_FLD_URL: img_content,
                            c.LLM_FLD_DETAIL: fidelity,
                        },
                    },
                ],
            },
        ]
        response = self.vlm_client.chat.completions.create(
            model=cred.AZURE_VLM_MODEL,
            messages=messages,
            response_format={"type": response_format},
            temperature=temperature,
        )
        self.usage_data[cred.AZURE_VLM_MODEL][
            c.FLD_USAGE_PROMPT
        ] += response.usage.prompt_tokens
        self.usage_data[cred.AZURE_VLM_MODEL][
            c.FLD_USAGE_COMPLETION
        ] += response.usage.completion_tokens
        dump_usage_data(self.usage_data, self.usage_file)
        return response.choices[0].message.content

    def describe(
        self,
        path: str,
        scene: ClipScene,
        abort_at_unknown: bool = False,
        unknown_desc: Optional[str] = None,
        fidelity: Literal["low", "high"] = "auto",
    ) -> Optional[dict[str, str]]:
        if scene in self.scene_to_desc:
            return self.scene_to_desc[scene]

        if scene == ClipScene.DIAGRAM:
            sys_msg = sys_msg_dsc_diagram
        elif scene == ClipScene.TEXT:
            sys_msg = sys_msg_desc_text
        else:
            if abort_at_unknown:
                return unknown_desc
            sys_msg = sys_msg_dsc_entities

        img_desc = self.call_4o(
            sys_msg, stringify_image(path), fidelity=fidelity
        )
        data = json.loads(img_desc)
        if type(data) is dict and "description" in data and "keywords" in data:
            return data
        logger.error(f"invalid image description, missing keys: {img_desc}")

    def process(
        self,
        path: str,
        no_caption: bool = False,
        scene: Optional[ClipScene] = None,
    ) -> Image:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        metadata = get_file_metadata(path)
        if not metadata["mime_type"].startswith("image/"):
            raise ValueError(f"Doesn't seem to be an image {path}")

        scene = scene or self.classify(path)
        shape = cv2.imread(path).shape[:2]

        # Check global no-caption mode
        if no_caption:
            return Image(
                **metadata,
                shape=f"{shape[0]}x{shape[1]}",
                scene=scene,
                description="",  # Placeholder for batch processing
                keywords=[],  # Placeholder for batch processing
            )

        desc_dict = self.describe(path, scene)
        if desc_dict is None:
            raise RuntimeError(f"Failed to describe image: {path}")

        return Image(
            **metadata,
            shape=f"{shape[0]}x{shape[1]}",
            scene=scene,
            description=desc_dict["description"],
            keywords=desc_dict["keywords"],
        )
