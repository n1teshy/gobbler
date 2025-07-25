from typing import Optional

from agentic_doc.common import ChunkType

import core.db.utils as db_utils
from core.processors.docs.core import DocumentProcessor
from core.processors.docs.models import DocumentObject
from core.processors.docs.utils import chunk_type_to_idx, idx_to_chunk_type
from core.processors.image.core import ImageProcessor
from core.processors.image.models import Image
from core.processors.image.utils import SceneType, idx_to_scene_type, scene_type_to_idx
from core.processors.video.core import VideoProcessor
from core.processors.video.models import Span
from core.utils import hash_file

_image_processor: ImageProcessor | None = None
_video_processor: VideoProcessor | None = None
_doc_processor: DocumentProcessor | None = None


# --- image functions ---


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
    Ingest an image file into the database.

    Parameters:
        path (str): Path to the image file.
        uploaded_by (str): User who uploaded the image. Defaults to 'system'.
        version (Optional[float]): Version number (defaults to file modification time).
        scene (Optional[SceneType]): Scene type (computed using CLIP if not provided).
        span_id (Optional[int]): Optional span ID to link image to.
        keywords (list[str], optional): List of keywords to associate with the image.
        throw_if_duplicate (bool): If True, raises error if image with same hash exists. Defaults to True.

    Returns:
        Image: Image object with database ID and updated attributes.
    """
    if throw_if_duplicate:
        existing_image = db_utils.images_collection.query(
            expr=f'hash == "{hash_file(path)}"', output_fields=["id"], limit=1
        )
        if existing_image:
            raise ValueError("this image already exists")

    global _image_processor
    _image_processor = _image_processor or ImageProcessor()
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
            "scene": processed_image.scene and scene_type_to_idx(processed_image.scene),
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


def search_images(
    query: Optional[str] = None,
    mime_type: Optional[str] = None,
    uploaded_by: Optional[str] = None,
    hash: Optional[str] = None,
    scene: Optional[SceneType] = None,
    keywords: Optional[list[str]] = None,
    span_id: Optional[int] = None,
    uploaded_before: Optional[float] = None,
    uploaded_after: Optional[float] = None,
    limit: int = 10,
    skip: int = 0,
) -> list[Image]:
    """
    Search for images in the database using metadata and/or vector search.

    Parameters:
        mime_type (Optional[str]): Filter by MIME type.
        uploaded_by (Optional[str]): Filter by uploader.
        hash (Optional[str]): Filter by file hash.
        description (Optional[str]): Search by description (vector search if provided).
        keywords (Optional[list[str]]): Filter by keywords.
        span_id (Optional[int]): Filter by associated span ID.
        uploaded_before (Optional[float]): Filter by upload time (before).
        uploaded_after (Optional[float]): Filter by upload time (after).
        limit (int): Maximum number of results to return. Defaults to 10.
        skip (int): Number of rows to skip (only for non-vector search). Defaults to 0.

    Returns:
        list[Image]: List of Image objects matching the query.
    """
    if query is not None and skip > 0:
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
    if scene is not None:
        exprs.append(f"scene == {scene_type_to_idx(scene)}")
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
    if query is not None:
        vector = db_utils.embedder.embed([query])[0]
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
        return [
            Image(
                **(
                    {hit.entity["entity"]}
                    | {"scene": idx_to_scene_type(hit.entity["entity"]["scene"])}
                )
            )
            for hit in hits
            if "entity" in hit.entity
        ]
    else:
        results = db_utils.images_collection.query(
            expr=expr, output_fields=output_fields, limit=limit, offset=skip
        )
        return [
            Image(**(r | {"scene": idx_to_scene_type(r["scene"])})) for r in results
        ]


# --- video functions ---


def ingest_video(
    path: str,
    uploaded_by: str = "system",
    version: Optional[float] = None,
    throw_if_duplicate: bool = True,
) -> list[Span]:
    """
    Ingest a video file into the database with atomic transaction.

    Parameters:
        path (str): Path to the video file.
        uploaded_by (str): User who uploaded the video. Defaults to 'system'.
        version (Optional[float]): Version number (defaults to file modification time).
        throw_if_duplicate (bool): If True, raises error if video with same hash exists. Defaults to True.

    Returns:
        list[Span]: List of Span objects with database IDs and updated attributes.
    """
    if throw_if_duplicate:
        existing_image = db_utils.spans_collection.query(
            expr=f'hash == "{hash_file(path)}"', output_fields=["id"], limit=1
        )
        if existing_image:
            raise ValueError("this video already exists")

    global _video_processor
    _video_processor = _video_processor or VideoProcessor()
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


def search_spans(
    short_query: Optional[str] = None,
    long_query: Optional[str] = None,
    video_id: Optional[int] = None,
    keywords: Optional[list[str]] = None,
    uploaded_before: Optional[float] = None,
    uploaded_after: Optional[float] = None,
    limit: int = 10,
    skip: int = 0,
) -> list[Span]:
    """
    Search for video spans in the database using metadata and/or vector search.

    Parameters:
        video_id (Optional[int]): Filter by video ID.
        short_description (Optional[str]): Search by short description (vector search if provided).
        long_description (Optional[str]): Search by long description (vector search if provided).
        keywords (Optional[list[str]]): Filter by keywords.
        uploaded_before (Optional[float]): Filter by upload time (before).
        uploaded_after (Optional[float]): Filter by upload time (after).
        limit (int): Maximum number of results to return. Defaults to 10.
        skip (int): Number of rows to skip (only for non-vector search). Defaults to 0.

    Returns:
        list[Span]: List of Span objects matching the query.
    """
    if short_query is not None or long_query is not None and skip > 0:
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
    if short_query is not None:
        vector = db_utils.embedder.embed([short_query])[0]
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
    elif long_query is not None:
        vector = db_utils.embedder.embed([long_query])[0]
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


# --- document functions ---


def ingest_document(
    path: str,
    uploaded_by: str = "system",
    version: Optional[float] = None,
    throw_if_duplicate: bool = True,
) -> list[DocumentObject]:
    """
    Parameters:
        path (str): Path to the document file.
        uploaded_by (str): User who uploaded the document. Defaults to
            'system'.
        version (Optional[float]): Version number (defaults to file
            modification time).
        throw_if_duplicate (bool): If True, raises error if document with
            same hash exists. Defaults to True.

    Returns:
        list[DocumentObject]: List of DocumentObject instances.
    """
    if throw_if_duplicate:
        existing_doc = db_utils.doc_obj_collection.query(
            expr=f'hash == "{hash_file(path)}"', output_fields=["id"], limit=1
        )
        if existing_doc:
            raise ValueError("this document already exists")

    global _doc_processor
    _doc_processor = _doc_processor or DocumentProcessor()
    inserted_ids = []

    try:
        doc_objects = _doc_processor.process(path)

        for doc_obj in doc_objects:
            if uploaded_by is not None:
                doc_obj.uploaded_by = uploaded_by
            if version is not None:
                doc_obj.version = version

            doc_data = {
                "URI": doc_obj.URI,
                "mime_type": doc_obj.mime_type,
                "size": doc_obj.size,
                "uploaded_by": doc_obj.uploaded_by,
                "uploaded_at": doc_obj.uploaded_at,
                "version": doc_obj.version,
                "hash": doc_obj.hash,
                "page": doc_obj.page,
                "position": list(doc_obj.position),
                "type": chunk_type_to_idx(doc_obj.type),
                "content": doc_obj.content,
                "content_vector": db_utils.embedder.embed([doc_obj.content])[0],
                "keywords": doc_obj.keywords,
            }
            result = db_utils.doc_obj_collection.insert(data=[doc_data])
            db_utils.doc_obj_collection.flush()
            doc_obj_id = result.primary_keys[0]
            doc_obj.id = doc_obj_id
            inserted_ids.append(doc_obj_id)
        return doc_objects
    except Exception as e:
        for doc_obj_id in inserted_ids:
            try:
                db_utils.images_collection.delete(expr=f"id == {doc_obj_id}")
                db_utils.images_collection.flush()
            except Exception:
                pass
        raise RuntimeError(f"Failed to process document: {str(e)}")


def search_document_objects(
    query: Optional[str] = None,
    page: Optional[int] = None,
    type: Optional[ChunkType] = None,
    keywords: Optional[list[str]] = None,
    limit: int = 10,
    skip: int = 0,
) -> list[DocumentObject]:
    """
    Search for document objects in the database using metadata and/or vector search.

    Parameters:
        page (Optional[int]): Filter by page number.
        type (Optional[ChunkType]): Filter by chunk type.
        keywords (Optional[list[str]]): Filter by keywords.
        content (Optional[str]): Search by content (vector search if provided).
        limit (int): Maximum number of results to return. Defaults to 10.
        skip (int): Number of rows to skip (only for non-vector search). Defaults to 0.

    Returns:
        list[DocumentObject]: List of DocumentObject instances matching the query.
    """
    if query is not None and skip > 0:
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
        "page",
        "position",
        "type",
        "content",
        "keywords",
    ]

    exprs = []
    if page is not None:
        exprs.append(f"page == {page}")
    if type is not None:
        exprs.append(f"type == {chunk_type_to_idx(type)}")
    if keywords:
        for kw in keywords:
            exprs.append(f'JSON_CONTAINS(keywords, "{kw}")')

    expr = " and ".join(exprs) if exprs else ""
    if query is not None:
        vector = db_utils.embedder.embed([query])[0]
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = db_utils.doc_obj_collection.search(
            data=[vector],
            anns_field="content_vector",
            param=search_params,
            limit=limit,
            expr=expr or None,
            output_fields=output_fields,
        )
        hits = results[0] if results else []
        return [
            DocumentObject(
                **(
                    hit.entity["entity"]
                    | {"type": idx_to_chunk_type(hit.entity["entity"]["type"])}
                )
            )
            for hit in hits
            if "entity" in hit.entity
        ]
    else:
        results = db_utils.doc_obj_collection.query(
            expr=expr, output_fields=output_fields, limit=limit, offset=skip
        )
        return [
            DocumentObject(**(r | {"type": idx_to_chunk_type(r["type"])}))
            for r in results
        ]
