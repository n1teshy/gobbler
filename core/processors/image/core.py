import base64
import json
import mimetypes
import os
from typing import Literal, Optional

import cv2
from openai import AzureOpenAI

import core.constants as c
import core.cred as cred
import core.globals as glb
from core.processors.image.models import Image
from core.processors.image.utils import (
    SceneType,
    classify_images,
    sys_msg_desc_text,
    sys_msg_dsc_diagram,
    sys_msg_dsc_entities,
)
from core.processors.interfaces import BaseProcessor
from core.utils import (
    dump_usage_data,
    get_file_metadata,
    get_usage_file,
    load_usage_data,
)


class ImageProcessor(BaseProcessor):
    def __init__(self, scene_to_desc: Optional[dict[SceneType, str]] = None):
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
        self.usage_data = load_usage_data(self.usage_file)
        self.usage_data[cred.AZURE_VLM_MODEL] = self.usage_data.get(
            cred.AZURE_VLM_MODEL,
            {c.FLD_USAGE_PROMPT: 0, c.FLD_USAGE_COMPLETION: 0},
        )

    def classify(self, path: str) -> Optional[SceneType]:
        scene, prob = classify_images([path])[0][0]
        return scene if prob >= glb.clip_prob_thresh else None

    def describe(
        self,
        path: str,
        scene: SceneType,
        mime: str,
        abort_at_unknown: bool = False,
        unknown_desc: Optional[str] = None,
        fidelity: Literal["low", "high"] = "auto",
    ) -> Optional[dict[str, str]]:
        if scene in self.scene_to_desc:
            return self.scene_to_desc[scene]

        if scene == SceneType.DIAGRAM:
            sys_msg = sys_msg_dsc_diagram
        elif scene == SceneType.TEXT:
            sys_msg = sys_msg_desc_text
        else:
            if abort_at_unknown:
                return unknown_desc
            sys_msg = sys_msg_dsc_entities

        b64_image = base64.b64encode(open(path, "rb").read()).decode("utf-8")
        for _ in range(3):
            response = self.vlm_client.chat.completions.create(
                model=cred.AZURE_VLM_MODEL,
                messages=[
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
                                    c.LLM_FLD_URL: f"data:{mime};base64,{b64_image}",
                                    c.LLM_FLD_DETAIL: fidelity,
                                },
                            },
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.5,
            )
            self.usage_data[cred.AZURE_VLM_MODEL][
                c.FLD_USAGE_PROMPT
            ] += response.usage.prompt_tokens
            self.usage_data[cred.AZURE_VLM_MODEL][
                c.FLD_USAGE_COMPLETION
            ] += response.usage.completion_tokens
            dump_usage_data(self.usage_data, self.usage_file)

            data = json.loads(response.choices[0].message.content)
            if (
                type(data) is dict
                and "description" in data
                and "keywords" in data
            ):
                return data

    def process(self, path: str, scene: Optional[SceneType] = None) -> Image:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        typ, _ = mimetypes.guess_type(path)
        if not typ.startswith("image/"):
            raise ValueError(f"Unsupported file: {path}")

        scene = scene or self.classify(path)
        shape = cv2.imread(path).shape[:2]
        desc_dict = self.describe(path, scene, typ)
        if desc_dict is None:
            raise RuntimeError(f"Failed to describe image: {path}")

        metadata = get_file_metadata(path)
        if not metadata["mime_type"].startswith("image/"):
            raise ValueError(f"Doesn't seem to be an image {path}")
        return Image(
            **metadata,
            shape=f"{shape[0]}x{shape[1]}",
            scene=scene,
            description=desc_dict["description"],
            keywords=desc_dict["keywords"],
        )
