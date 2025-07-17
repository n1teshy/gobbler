import base64
import mimetypes
from typing import Literal, Optional

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


class ImageProcessor(BaseProcessor):
    def __init__(self, scene_to_desc: Optional[dict[SceneType, str]] = None):
        if not cred.AZURE_VLM_KEY:
            raise EnvironmentError("Missing Azure VLM key")
        self.vlm_client = AzureOpenAI(
            api_key=cred.AZURE_VLM_KEY,
            azure_endpoint=cred.AZURE_VLM_BASE,
            azure_deployment=cred.AZURE_VLM_DEPLOYMENT,
            api_version=cred.AZURE_VLM_VERSION,
        )
        self.scene_to_desc = scene_to_desc or {}

    def classify(self, path: str) -> Optional[SceneType]:
        scene, prob = classify_images([path])[0][0]
        return scene if prob >= glb.clip_prob_thresh else None

    def describe(
        self,
        path: str,
        scene: SceneType,
        abort_at_unknown: bool = False,
        unknown_desc: Optional[str] = None,
        fidelity: Literal["low", "high"] = "auto",
    ) -> Optional[str]:
        typ, _ = mimetypes.guess_type(path)
        if not typ.startswith("image/"):
            raise ValueError(f"Unsupported file: {path}")

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
                                c.LLM_FLD_URL: f"data:{typ};base64,{b64_image}",
                                c.LLM_FLD_DETAIL: fidelity,
                            },
                        },
                    ],
                },
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()

    def process(self, path: str, scene: Optional[SceneType] = None) -> Image:
        scene = scene or self.classify(path)
        description = self.describe(path, scene)
        return Image(
            URI=path,
            scene=scene,
            description=description,
        )
