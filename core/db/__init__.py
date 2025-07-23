from typing import Optional

import core.db.utils as db_utils
from core.processors.image.core import ImageProcessor
from core.processors.image.models import Image
from core.processors.image.utils import SceneType
from core.processors.video.core import VideoProcessor
from core.processors.video.models import Span
from core.utils import hash_file

_image_processor = ImageProcessor()
_video_processor = VideoProcessor()


def ingest_image(
    path: str,
    uploaded_by: str = "system",
    version: Optional[float] = None,
    scene: Optional[SceneType] = None,
    span_id: Optional[int] = None,
    keywords: list[str] = None,
    throw_if_duplicate: bool = True,
) -> Image:
    """
    Ingest an image file into the database

    Args:
        image_path: Path to the image file
        uploaded_by: User who uploaded the image
        version: Version number (defaults to file modification time)
        scene: Scene type (computed using CLIP if not provided)
        span_id: Optional span ID to link image to

    Returns:
        Image object with database ID and updated attributes
    """
    if throw_if_duplicate:
        existing_image = db_utils.images_collection.query(
            expr=f'hash == "{hash_file(path)}"', output_fields=["id"], limit=1
        )
        if existing_image:
            raise ValueError("this image already exists")

    processed_image = _image_processor.process(path, scene)
    if uploaded_by is not None:
        processed_image.uploaded_by = uploaded_by
    if version is not None:
        processed_image.version = version
    if keywords is not None:
        processed_image.keywords = keywords

    try:
        image_data = {
            "URI": processed_image.URI,
            "mime_type": processed_image.mime_type,
            "size": processed_image.size,
            "uploaded_by": processed_image.uploaded_by,
            "uploaded_at": processed_image.uploaded_at,
            "version": processed_image.version,
            "hash": processed_image.hash,
            "shape": processed_image.shape,
            "scene": processed_image.scene.value if processed_image.scene else None,
            "description": processed_image.description,
            "keywords": processed_image.keywords,
            "description_vector": db_utils.embedder.embed(
                [processed_image.description]
            )[0],
            "span_id": span_id,
        }
        result = db_utils.images_collection.insert(data=[image_data])
        db_utils.images_collection.flush()
        image_id = result.primary_keys[0]
        processed_image.id = image_id
        return processed_image
    except Exception as e:
        raise RuntimeError(f"Failed to process image: {str(e)}")


def ingest_video(
    path: str,
    uploaded_by: str = "system",
    version: Optional[float] = None,
    throw_if_duplicate: bool = True,
) -> list[Span]:
    """
    Ingest a video file into the database with atomic transaction

    Args:
        video_path: Path to the video file
        uploaded_by: User who uploaded the video
        version: Version number (defaults to file modification time)

    Returns:
        Video object with database ID and updated attributes
    """
    if throw_if_duplicate:
        existing_image = db_utils.spans_collection.query(
            expr=f'hash == "{hash_file(path)}"', output_fields=["id"], limit=1
        )
        if existing_image:
            raise ValueError("this video already exists")

    spans = _video_processor.process(path)
    inserted_span_ids = []
    inserted_image_ids = []

    try:
        for span in spans:
            if uploaded_by is not None:
                span.uploaded_by = uploaded_by
            if version is not None:
                span.version = version
            span_data = {
                "URI": span.URI,
                "mime_type": span.mime_type,
                "size": span.size,
                "uploaded_by": span.uploaded_by,
                "uploaded_at": span.uploaded_at,
                "version": span.version,
                "hash": span.hash,
                "start": span.start,
                "end": span.end,
                "duration": span.end - span.start,
                "short_description": span.short_description,
                "long_description": span.long_description,
                "keywords": span.keywords,
                "short_description_vector": db_utils.embedder.embed(
                    [span.short_description]
                )[0],
                "long_description_vector": db_utils.embedder.embed(
                    [span.long_description]
                )[0],
            }
            span_result = db_utils.spans_collection.insert(data=[span_data])
            db_utils.spans_collection.flush()
            span_id = span_result.primary_keys[0]
            inserted_span_ids.append(span_id)
            span.id = span_id

            for frame in span.frames:
                frame_data = {
                    "URI": frame.URI,
                    "mime_type": frame.mime_type,
                    "size": frame.size,
                    "uploaded_by": frame.uploaded_by,
                    "uploaded_at": frame.uploaded_at,
                    "version": frame.version,
                    "hash": frame.hash,
                    "shape": frame.shape,
                    "scene": frame.scene.value if frame.scene else None,
                    "description": frame.description,
                    "keywords": frame.keywords,
                    "description_vector": db_utils.embedder.embed([frame.description])[
                        0
                    ],
                    "span_id": span_id,
                }
                frame_result = db_utils.images_collection.insert(data=[frame_data])
                db_utils.images_collection.flush()
                frame_id = frame_result.primary_keys[0]
                inserted_image_ids.append(frame_id)
        return spans
    except Exception as e:
        for image_id in inserted_image_ids:
            try:
                db_utils.images_collection.delete(expr=f"id == {image_id}")
                db_utils.images_collection.flush()
            except Exception:
                pass
        for span_id in inserted_span_ids:
            try:
                db_utils.spans_collection.delete(expr=f"id == {span_id}")
                db_utils.spans_collection.flush()
            except Exception:
                pass
        raise RuntimeError(f"Failed to process video: {str(e)}")


def search_images(
    mime_type: Optional[str] = None,
    uploaded_by: Optional[str] = None,
    hash: Optional[str] = None,
    description: Optional[str] = None,
    keywords: Optional[list[str]] = None,
    span_id: Optional[int] = None,
    uploaded_before: Optional[float] = None,
    uploaded_after: Optional[float] = None,
    limit: int = 10,
    skip: int = 0,
) -> list[Image]:
    if description is not None and skip > 0:
        raise ValueError("row-skipping is not supported for vector search")

    output_fields = [
        "id",
        "URI",
        "mime_type",
        "size",
        "uploaded_by",
        "uploaded_at",
        "version",
        "hash",
        "shape",
        "scene",
        "description",
        "keywords",
        "span_id",
    ]

    exprs = []
    if mime_type:
        exprs.append(f'mime_type == "{mime_type}"')
    if uploaded_by:
        exprs.append(f'uploaded_by == "{uploaded_by}"')
    if hash:
        exprs.append(f'hash == "{hash}"')
    if span_id is not None:
        exprs.append(f"span_id == {span_id}")
    if uploaded_before is not None:
        exprs.append(f"uploaded_at < {uploaded_before}")
    if uploaded_after is not None:
        exprs.append(f"uploaded_at > {uploaded_after}")
    if keywords:
        for kw in keywords:
            exprs.append(f'JSON_CONTAINS(keywords, "{kw}")')
    expr = " and ".join(exprs) if exprs else ""
    if description is not None:
        vector = db_utils.embedder.embed([description])[0]
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = db_utils.images_collection.search(
            data=[vector],
            anns_field="description_vector",
            param=search_params,
            limit=limit,
            expr=expr or None,
            output_fields=output_fields,
        )
        hits = results[0] if results else []
        print([hit.entity for hit in hits])
        return [Image(**hit.entity["entity"]) for hit in hits if "entity" in hit.entity]
    else:
        results = db_utils.images_collection.query(
            expr=expr, output_fields=output_fields, limit=limit, offset=skip
        )
        return [Image(**r) for r in results]


def search_spans(
    video_id: Optional[int] = None,
    short_description: Optional[str] = None,
    long_description: Optional[str] = None,
    keywords: Optional[list[str]] = None,
    uploaded_before: Optional[float] = None,
    uploaded_after: Optional[float] = None,
    limit: int = 10,
    skip: int = 0,
) -> list[Span]:
    if short_description is not None or long_description is not None and skip > 0:
        raise ValueError("row-skipping is not supported for vector search")

    output_fields = [
        "id",
        "URI",
        "mime_type",
        "size",
        "uploaded_by",
        "uploaded_at",
        "version",
        "hash",
        "start",
        "end",
        "duration",
        "short_description",
        "long_description",
        "keywords",
    ]

    exprs = []
    if video_id is not None:
        exprs.append(f"video_id == {video_id}")
    if keywords:
        for kw in keywords:
            exprs.append(f'JSON_CONTAINS(keywords, "{kw}")')
    if uploaded_before is not None:
        exprs.append(f"uploaded_at < {uploaded_before}")
    if uploaded_after is not None:
        exprs.append(f"uploaded_at > {uploaded_after}")
    expr = " and ".join(exprs) if exprs else ""
    if short_description is not None:
        vector = db_utils.embedder.embed([short_description])[0]
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = db_utils.spans_collection.search(
            data=[vector],
            anns_field="short_description_vector",
            param=search_params,
            limit=limit,
            expr=expr or None,
            output_fields=output_fields,
        )
        hits = results[0] if results else []
        return [Span(**hit.entity["entity"]) for hit in hits if "entity" in hit.entity]
    elif long_description is not None:
        vector = db_utils.embedder.embed([long_description])[0]
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = db_utils.spans_collection.search(
            data=[vector],
            anns_field="long_description_vector",
            param=search_params,
            limit=limit,
            expr=expr or None,
            output_fields=output_fields,
        )
        hits = results[0] if results else []
        return [Span(**hit.entity["entity"]) for hit in hits if "entity" in hit.entity]
    else:
        results = db_utils.spans_collection.query(
            expr=expr, output_fields=output_fields, limit=limit, offset=skip
        )
        return [Span(**r) for r in results]
