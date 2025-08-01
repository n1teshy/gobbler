import os
from typing import Any, Optional, Union

from agentic_doc.common import ChunkType

import gobbler.constants as c
import gobbler.db.utils as db_utils
from gobbler.processors.docs.core import DocumentProcessor
from gobbler.processors.docs.models import DocumentObject
from gobbler.processors.docs.utils import chunk_type_to_idx, idx_to_chunk_type
from gobbler.processors.image.core import ImageProcessor
from gobbler.processors.image.models import Image
from gobbler.processors.image.utils import (
    SceneType,
    idx_to_scene_type,
    scene_type_to_idx,
)
from gobbler.processors.video.core import VideoProcessor
from gobbler.processors.video.models import Span
from gobbler.utils import hash_file, uri_to_file

_image_processor: ImageProcessor | None = None
_video_processor: VideoProcessor | None = None
_doc_processor: DocumentProcessor | None = None


def get_image_processor() -> ImageProcessor:
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor()
    return _image_processor


def get_video_processor() -> VideoProcessor:
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor


def get_doc_processor() -> DocumentProcessor:
    global _doc_processor
    if _doc_processor is None:
        _doc_processor = DocumentProcessor()
    return _doc_processor


# --- image functions ---


def ingest_image(
    uri: str,
    extra_fields: Optional[dict[str, Any]] = None,
    uploaded_by: str = "system",
    version: Optional[float] = None,
    scene: Optional[SceneType] = None,
    span_id: Optional[int] = None,
    processor: Optional[ImageProcessor] = None,
    download_headers: Optional[dict[str, str]] = None,
    throw_if_duplicate: bool = True,
) -> Image:
    """
    Ingest an image file into the database.

    Parameters:
        uri (str): Path to the image file.
        extra_fields (Optional[dict[str, Any]]): Additional dynamic fields to
            store.
        uploaded_by (str): User who uploaded the image. Defaults to 'system'.
        version (Optional[float]): Version number (defaults to file
            modification time).
        scene (Optional[SceneType]): Scene type (computed using CLIP if not
            provided).
        span_id (Optional[int]): Optional span ID to link image to.
        processor (Optional[ImageProcessor]): Custom image processor instance.
            Defaults to None.
        download_headers (Optional[dict[str, str]]): Headers for downloading the
            URI content.
        throw_if_duplicate (bool): If True, raises error if image with same
            hash exists. Defaults to True.

    Returns:
        Image: Image object with database ID and updated attributes.
    """
    downloaded, path = uri_to_file(uri, headers=download_headers)
    try:
        if throw_if_duplicate:
            existing_image = db_utils.images_collection.query(
                expr=f'{c.DB_FLD_HASH} == "{hash_file(path)}"',
                output_fields=[c.DB_FLD_ID],
                limit=1,
            )
            if existing_image:
                raise ValueError("this image already exists")

        processor = processor or get_image_processor()
        processed_image = processor.process(path, scene)
        if downloaded:
            processed_image.uri = uri
        if uploaded_by is not None:
            processed_image.uploaded_by = uploaded_by
        if version is not None:
            processed_image.version = version

        image_data = {
            c.DB_FLD_URI: processed_image.uri,
            c.DB_FLD_MIME_TYPE: processed_image.mime_type,
            c.DB_FLD_SIZE: processed_image.size,
            c.DB_FLD_UPLOADED_BY: processed_image.uploaded_by,
            c.DB_FLD_UPLOADED_AT: processed_image.uploaded_at,
            c.DB_FLD_VERSION: processed_image.version,
            c.DB_FLD_HASH: processed_image.hash,
            c.IMG_FLD_SHAPE: processed_image.shape,
            c.IMG_FLD_SCENE: (
                scene_type_to_idx(processed_image.scene)
                if processed_image.scene
                else None
            ),
            c.IMG_FLD_DESCRIPTION: processed_image.description,
            c.DB_FLD_KEYWORDS: processed_image.keywords,
            c.IMG_FLD_DESCRIPTION_VECTOR: db_utils.embedder.embed(
                [processed_image.description]
            )[0],
            c.IMG_FLD_SPAN_ID: span_id,
        }
        if extra_fields:
            image_data.update(extra_fields)

        result = db_utils.images_collection.insert(data=[image_data])
        db_utils.images_collection.flush()
        image_id = result.primary_keys[0]
        processed_image.id = image_id
        if extra_fields is None:
            return processed_image
        return Image(**(processed_image.to_json() | extra_fields))
    except Exception as e:
        raise RuntimeError(f"Failed to process image: {str(e)}")
    finally:
        if downloaded:
            os.remove(path)


def search_images(
    query: Optional[Union[str, list[float]]] = None,
    mime_type: Optional[str] = None,
    uploaded_by: Optional[str] = None,
    scene: Optional[SceneType] = None,
    keywords: Optional[list[str]] = None,
    span_id: Optional[int] = None,
    uploaded_before: Optional[int] = None,
    uploaded_after: Optional[int] = None,
    extra_fields: Optional[dict[str, Any]] = None,
    limit: int = 10,
    skip: int = 0,
) -> list[tuple[float | None, Image]]:
    """
    Search for images in the database using metadata and/or vector search.

    Parameters:
        query (Optional[str | list[float]]): Natural language query or
            vector representation of it for cosine similarity search.
        mime_type (Optional[str | list[float]]): Filter by MIME type.
        uploaded_by (Optional[str]): Filter by uploader.
        description (Optional[str]): Search by description (vector search if
            provided).
        keywords (Optional[list[str]]): Filter by keywords.
        span_id (Optional[int]): Filter by associated span ID.
        uploaded_before (Optional[int]): Filter by upload time (before).
        uploaded_after (Optional[int]): Filter by upload time (after).
        extra_fields (Optional[dict[str, Any]]): Additional dynamic fields to
            filter by.
        limit (int): Maximum number of results to return. Defaults to 10.
        skip (int): Number of rows to skip (only for non-vector search).
            Defaults to 0.

    Returns:
        list[Image]: List of Image objects matching the query.
    """
    if query is not None and skip > 0:
        raise ValueError("row-skipping is not supported for vector search")

    output_fields = [
        c.DB_FLD_ID,
        c.DB_FLD_URI,
        c.DB_FLD_MIME_TYPE,
        c.DB_FLD_SIZE,
        c.DB_FLD_UPLOADED_BY,
        c.DB_FLD_UPLOADED_AT,
        c.DB_FLD_VERSION,
        c.DB_FLD_HASH,
        c.IMG_FLD_SHAPE,
        c.IMG_FLD_SCENE,
        c.IMG_FLD_DESCRIPTION,
        c.DB_FLD_KEYWORDS,
        c.IMG_FLD_SPAN_ID,
    ]
    if extra_fields:
        output_fields.extend(extra_fields.keys())

    exprs = []
    images = []

    if mime_type:
        exprs.append(f'{c.DB_FLD_MIME_TYPE} == "{mime_type}"')
    if uploaded_by:
        exprs.append(f'{c.DB_FLD_UPLOADED_BY} == "{uploaded_by}"')
    if scene is not None:
        exprs.append(f"{c.IMG_FLD_SCENE} == {scene_type_to_idx(scene)}")
    if span_id is not None:
        exprs.append(f"{c.IMG_FLD_SPAN_ID} == {span_id}")
    if uploaded_before is not None:
        exprs.append(f"{c.DB_FLD_UPLOADED_AT} < {uploaded_before}")
    if uploaded_after is not None:
        exprs.append(f"{c.DB_FLD_UPLOADED_AT} > {uploaded_after}")
    if keywords:
        for kw in keywords:
            exprs.append(f'JSON_CONTAINS({c.DB_FLD_KEYWORDS}, "{kw}")')
    if extra_fields:
        for field, value in extra_fields.items():
            exprs.append(f'{field} == "{value}"')
    expr = " and ".join(exprs) if exprs else ""

    if query is not None:
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = db_utils.images_collection.search(
            data=[
                (
                    db_utils.embedder.embed([query])[0]
                    if type(query) is str
                    else query
                )
            ],
            anns_field=c.IMG_FLD_DESCRIPTION_VECTOR,
            param=search_params,
            limit=limit,
            expr=expr or None,
            output_fields=output_fields,
        )

        for hit in results[0] if results else []:
            if "entity" not in hit.entity:
                continue
            hit_data = hit.entity["entity"]
            if hit_data[c.IMG_FLD_SCENE] is not None:
                hit_data[c.IMG_FLD_SCENE] = idx_to_scene_type(
                    hit_data[c.IMG_FLD_SCENE]
                )
            images.append((hit.distance, Image(**hit_data)))
    else:
        hits = db_utils.images_collection.query(
            expr=expr, output_fields=output_fields, limit=limit, offset=skip
        )
        for hit in hits:
            if hit[c.IMG_FLD_SCENE] is not None:
                hit[c.IMG_FLD_SCENE] = idx_to_scene_type(hit[c.IMG_FLD_SCENE])
            images.append((None, Image(**hit)))

    return images


# --- video functions ---


def ingest_video(
    uri: str,
    extra_fields: Optional[dict[str, Any]] = None,
    extra_frame_fields: Optional[dict[str, Any]] = None,
    uploaded_by: str = "system",
    version: Optional[float] = None,
    processor: Optional[VideoProcessor] = None,
    audio_only: bool = False,
    download_headers: Optional[dict[str, str]] = None,
    throw_if_duplicate: bool = True,
) -> list[Span]:
    """
    Ingest a video file into the database with atomic transaction.

    Parameters:
        uri (str): Path to the video file.
        extra_fields (Optional[dict[str, Any]]): Additional dynamic fields to
            store.
        extra_frame_fields (Optional[dict[str, Any]]): Additional dynamic fields
            to store for each frame.
        uploaded_by (str): User who uploaded the video. Defaults to 'system'.
        version (Optional[float]): Version number (defaults to file modification
            time).
        processor (Optional[VideoProcessor]): Custom video processor instance.
            Defaults to None.
        audio_only (bool): If True, processes only the audio track. Defaults to
            False.
        download_headers (Optional[dict[str, str]]): Headers for downloading the
            URI content.
        throw_if_duplicate (bool): If True, raises error if video with same
            hash exists. Defaults to True.

    Returns:
        list[Span]: List of Span objects with database IDs and updated
            attributes.
    """
    downloaded, path = uri_to_file(uri, headers=download_headers)
    inserted_span_ids = []
    inserted_image_ids = []

    try:
        if throw_if_duplicate:
            existing_image = db_utils.spans_collection.query(
                expr=f'{c.DB_FLD_HASH} == "{hash_file(path)}"',
                output_fields=[c.DB_FLD_ID],
                limit=1,
            )
            if existing_image:
                raise ValueError("this video already exists")

        processor = processor or get_video_processor()
        spans = processor.process(path, audio_only=audio_only)
        if downloaded:
            for span in spans:
                span.uri = uri

        for span_idx in range(len(spans)):
            span = spans[span_idx]
            if uploaded_by is not None:
                span.uploaded_by = uploaded_by
            if version is not None:
                span.version = version
            span_data = {
                c.DB_FLD_URI: span.uri,
                c.DB_FLD_MIME_TYPE: span.mime_type,
                c.DB_FLD_SIZE: span.size,
                c.DB_FLD_UPLOADED_BY: span.uploaded_by,
                c.DB_FLD_UPLOADED_AT: span.uploaded_at,
                c.DB_FLD_VERSION: span.version,
                c.DB_FLD_HASH: span.hash,
                c.SPAN_FLD_START: span.start,
                c.SPAN_FLD_END: span.end,
                c.SPAN_FLD_SHORT_DESCRIPTION: span.short_description,
                c.SPAN_FLD_LONG_DESCRIPTION: span.long_description,
                c.DB_FLD_KEYWORDS: span.keywords,
                c.SPAN_FLD_SHORT_DESCRIPTION_VECTOR: db_utils.embedder.embed(
                    [span.short_description]
                )[0],
                c.SPAN_FLD_LONG_DESCRIPTION_VECTOR: db_utils.embedder.embed(
                    [span.long_description]
                )[0],
            }
            if extra_fields:
                span_data.update(extra_fields)
            span_result = db_utils.spans_collection.insert(data=[span_data])
            db_utils.spans_collection.flush()
            span_id = span_result.primary_keys[0]
            inserted_span_ids.append(span_id)
            span.id = span_id
            if extra_fields is not None:
                spans[span_idx] = Span(**(span.to_json() | extra_fields))

            for frame in span.frames:
                frame_data = {
                    c.DB_FLD_URI: frame.uri,
                    c.DB_FLD_MIME_TYPE: frame.mime_type,
                    c.DB_FLD_SIZE: frame.size,
                    c.DB_FLD_UPLOADED_BY: frame.uploaded_by,
                    c.DB_FLD_UPLOADED_AT: frame.uploaded_at,
                    c.DB_FLD_VERSION: frame.version,
                    c.DB_FLD_HASH: frame.hash,
                    c.IMG_FLD_SHAPE: frame.shape,
                    c.IMG_FLD_SCENE: (
                        frame.scene.value if frame.scene else None
                    ),
                    c.IMG_FLD_DESCRIPTION: frame.description,
                    c.DB_FLD_KEYWORDS: frame.keywords,
                    c.IMG_FLD_DESCRIPTION_VECTOR: db_utils.embedder.embed(
                        [frame.description]
                    )[0],
                    c.IMG_FLD_SPAN_ID: span_id,
                }
                if extra_frame_fields:
                    frame_data.update(extra_frame_fields)
                frame_result = db_utils.images_collection.insert(
                    data=[frame_data]
                )
                db_utils.images_collection.flush()
                frame_id = frame_result.primary_keys[0]
                inserted_image_ids.append(frame_id)
        return spans
    except Exception as e:
        for image_id in inserted_image_ids:
            try:
                db_utils.images_collection.delete(
                    expr=f"{c.DB_FLD_ID} == {image_id}"
                )
                db_utils.images_collection.flush()
            except Exception:
                pass
        for span_id in inserted_span_ids:
            try:
                db_utils.spans_collection.delete(
                    expr=f"{c.DB_FLD_ID} == {span_id}"
                )
                db_utils.spans_collection.flush()
            except Exception:
                pass
        raise RuntimeError(f"Failed to process video: {str(e)}")
    finally:
        if downloaded:
            os.remove(path)


def search_spans(
    short_query: Optional[Union[str, list[float]]] = None,
    long_query: Optional[Union[str, list[float]]] = None,
    mime_type: Optional[str] = None,
    keywords: Optional[list[str]] = None,
    uploaded_before: Optional[int] = None,
    uploaded_after: Optional[int] = None,
    extra_fields: Optional[dict[str, Any]] = None,
    limit: int = 10,
    skip: int = 0,
) -> list[tuple[float | None, Span]]:
    """
    Search for video spans in the database using metadata and/or vector search.

    Parameters:
        short_query (Optional[str | list[float]]): Natural language query or
            vector representation of it to match against span's
            `short_description`.
        long_query (Optional[str | list[float]]): Natural language query or
            vector representation of it to match against span's
            `long_description`.
        mime_type (Optional[str]): Filter by MIME type.
        keywords (Optional[list[str]]): Filter by keywords.
        uploaded_before (Optional[int]): Filter by upload time (before).
        uploaded_after (Optional[int]): Filter by upload time (after).
        extra_fields (Optional[dict[str, Any]]): Additional dynamic fields to
            filter by.
        limit (int): Maximum number of results to return. Defaults to 10.
        skip (int): Number of rows to skip (only for non-vector search).
            Defaults to 0.

    Returns:
        list[Span]: List of Span objects matching the query.
    """
    if short_query is not None or long_query is not None and skip > 0:
        raise ValueError("row-skipping is not supported for vector search")

    output_fields = [
        c.DB_FLD_ID,
        c.DB_FLD_URI,
        c.DB_FLD_MIME_TYPE,
        c.DB_FLD_SIZE,
        c.DB_FLD_UPLOADED_BY,
        c.DB_FLD_UPLOADED_AT,
        c.DB_FLD_VERSION,
        c.DB_FLD_HASH,
        c.SPAN_FLD_START,
        c.SPAN_FLD_END,
        c.SPAN_FLD_SHORT_DESCRIPTION,
        c.SPAN_FLD_LONG_DESCRIPTION,
        c.DB_FLD_KEYWORDS,
    ]
    if extra_fields:
        output_fields.extend(extra_fields.keys())

    exprs = []
    if mime_type:
        exprs.append(f'{c.DB_FLD_MIME_TYPE} == "{mime_type}"')
    if keywords:
        for kw in keywords:
            exprs.append(f'JSON_CONTAINS({c.DB_FLD_KEYWORDS}, "{kw}")')
    if uploaded_before is not None:
        exprs.append(f"{c.DB_FLD_UPLOADED_AT} < {uploaded_before}")
    if uploaded_after is not None:
        exprs.append(f"{c.DB_FLD_UPLOADED_AT} > {uploaded_after}")
    if extra_fields:
        for field, value in extra_fields.items():
            exprs.append(f'{field} == "{value}"')
    expr = " and ".join(exprs) if exprs else ""

    if short_query is not None:
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = db_utils.spans_collection.search(
            data=[
                (
                    db_utils.embedder.embed([short_query])[0]
                    if type(short_query) is str
                    else short_query
                )
            ],
            anns_field=c.SPAN_FLD_SHORT_DESCRIPTION_VECTOR,
            param=search_params,
            limit=limit,
            expr=expr or None,
            output_fields=output_fields,
        )
        hits = results[0] if results else []
        return [
            (hit.distance, Span(**hit.entity["entity"]))
            for hit in hits
            if "entity" in hit.entity
        ]
    elif long_query is not None:
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = db_utils.spans_collection.search(
            data=[
                (
                    db_utils.embedder.embed([long_query])[0]
                    if type(long_query) is str
                    else long_query
                )
            ],
            anns_field=c.SPAN_FLD_LONG_DESCRIPTION_VECTOR,
            param=search_params,
            limit=limit,
            expr=expr or None,
            output_fields=output_fields,
        )
        hits = results[0] if results else []
        return [
            (hit.distance, Span(**hit.entity["entity"]))
            for hit in hits
            if "entity" in hit.entity
        ]
    else:
        results = db_utils.spans_collection.query(
            expr=expr, output_fields=output_fields, limit=limit, offset=skip
        )
        return [(None, Span(**r)) for r in results]


# --- document functions ---


def ingest_document(
    uri: str,
    extra_fields: Optional[dict[str, Any]] = None,
    uploaded_by: str = "system",
    version: Optional[float] = None,
    processor: Optional[DocumentProcessor] = None,
    download_headers: Optional[dict[str, str]] = None,
    throw_if_duplicate: bool = True,
) -> list[DocumentObject]:
    """
    Parameters:
        uri (str): Path to the document file.
        extra_fields (Optional[dict[str, Any]]): Additional dynamic fields to
            store.
        uploaded_by (str): User who uploaded the document. Defaults to
            'system'.
        version (Optional[float]): Version number (defaults to file
            modification time).
        processor (Optional[DocumentProcessor]): Custom document processor
            instance. Defaults to None.
        download_headers (Optional[dict[str, str]]): Headers for downloading the
            URI content.
        throw_if_duplicate (bool): If True, raises error if document with
            same hash exists. Defaults to True.

    Returns:
        list[DocumentObject]: List of DocumentObject instances.
    """
    downloaded, path = uri_to_file(uri, headers=download_headers)
    inserted_ids = []

    try:
        if throw_if_duplicate:
            existing_doc = db_utils.doc_obj_collection.query(
                expr=f'{c.DB_FLD_HASH} == "{hash_file(path)}"',
                output_fields=[c.DB_FLD_ID],
                limit=1,
            )
            if existing_doc:
                raise ValueError("this document already exists")

        processor = processor or get_doc_processor()

        doc_objects = processor.process(path)
        if downloaded:
            for doc_obj in doc_objects:
                doc_obj.uri = uri

        for doc_obj_idx in range(len(doc_objects)):
            doc_obj = doc_objects[doc_obj_idx]
            if uploaded_by is not None:
                doc_obj.uploaded_by = uploaded_by
            if version is not None:
                doc_obj.version = version

            doc_data = {
                c.DB_FLD_URI: doc_obj.uri,
                c.DB_FLD_MIME_TYPE: doc_obj.mime_type,
                c.DB_FLD_SIZE: doc_obj.size,
                c.DB_FLD_UPLOADED_BY: doc_obj.uploaded_by,
                c.DB_FLD_UPLOADED_AT: doc_obj.uploaded_at,
                c.DB_FLD_VERSION: doc_obj.version,
                c.DB_FLD_HASH: doc_obj.hash,
                c.DOC_FLD_PAGE: doc_obj.page,
                c.DOC_FLD_POSITION: list(doc_obj.position),
                c.DOC_FLD_TYPE: chunk_type_to_idx(doc_obj.type),
                c.DOC_FLD_CONTENT: doc_obj.content,
                c.DOC_FLD_CONTENT_VECTOR: db_utils.embedder.embed(
                    [doc_obj.content]
                )[0],
                c.DB_FLD_KEYWORDS: doc_obj.keywords,
            }
            if extra_fields:
                doc_data.update(extra_fields)
            result = db_utils.doc_obj_collection.insert(data=[doc_data])
            db_utils.doc_obj_collection.flush()
            doc_obj_id = result.primary_keys[0]
            doc_obj.id = doc_obj_id
            if extra_fields is not None:
                doc_objects[doc_obj_idx] = DocumentObject(
                    **(doc_obj.to_json() | extra_fields)
                )
            inserted_ids.append(doc_obj_id)
        return doc_objects
    except Exception as e:
        for doc_obj_id in inserted_ids:
            try:
                db_utils.images_collection.delete(
                    expr=f"{c.DB_FLD_ID} == {doc_obj_id}"
                )
                db_utils.images_collection.flush()
            except Exception:
                pass
        raise e
    finally:
        if downloaded:
            os.remove(path)


def search_document_objects(
    query: Optional[str] = None,
    mime_type: Optional[str] = None,
    page: Optional[int] = None,
    chunk_type: Optional[ChunkType] = None,
    keywords: Optional[list[str]] = None,
    uploaded_before: Optional[int] = None,
    uploaded_after: Optional[int] = None,
    extra_fields: Optional[dict[str, Any]] = None,
    limit: int = 10,
    skip: int = 0,
) -> list[tuple[float | None, DocumentObject]]:
    """
    Search for document objects in the database using metadata and/or
        vector search.

    Parameters:
        query (Optional[str | list[float]]): Natural language query or
            vector representation of it for cosine similarity search.
        mime_type (Optional[str]): Filter by MIME type.
        page (Optional[int]): Filter by page number.
        chunk_type (Optional[ChunkType]): Filter by chunk type.
        keywords (Optional[list[str]]): Filter by keywords.
        uploaded_before (Optional[int]): Filter by upload time (before).
        uploaded_after (Optional[int]): Filter by upload time (after).
        content (Optional[str]): Search by content (vector search if provided).
        extra_fields (Optional[dict[str, Any]]): Additional dynamic fields to
            filter by.
        limit (int): Maximum number of results to return. Defaults to 10.
        skip (int): Number of rows to skip (only for non-vector search).
            Defaults to 0.

    Returns:
        list[DocumentObject]: List of DocumentObject instances matching the query.
    """
    if query is not None and skip > 0:
        raise ValueError("row-skipping is not supported for vector search")

    output_fields = [
        c.DB_FLD_ID,
        c.DB_FLD_URI,
        c.DB_FLD_MIME_TYPE,
        c.DB_FLD_SIZE,
        c.DB_FLD_UPLOADED_BY,
        c.DB_FLD_UPLOADED_AT,
        c.DB_FLD_VERSION,
        c.DB_FLD_HASH,
        c.DOC_FLD_PAGE,
        c.DOC_FLD_POSITION,
        c.DOC_FLD_TYPE,
        c.DOC_FLD_CONTENT,
        c.DB_FLD_KEYWORDS,
    ]
    if extra_fields:
        output_fields.extend(extra_fields.keys())

    exprs = []
    if page is not None:
        exprs.append(f"{c.DOC_FLD_PAGE} == {page}")
    if mime_type:
        exprs.append(f'{c.DB_FLD_MIME_TYPE} == "{mime_type}"')
    if chunk_type is not None:
        exprs.append(f"{c.DOC_FLD_TYPE} == {chunk_type_to_idx(chunk_type)}")
    if keywords:
        for kw in keywords:
            exprs.append(f'JSON_CONTAINS({c.DB_FLD_KEYWORDS}, "{kw}")')
    if uploaded_before is not None:
        exprs.append(f"{c.DB_FLD_UPLOADED_AT} < {uploaded_before}")
    if uploaded_after is not None:
        exprs.append(f"{c.DB_FLD_UPLOADED_AT} > {uploaded_after}")
    if extra_fields:
        for field, value in extra_fields.items():
            exprs.append(f'{field} == "{value}"')
    expr = " and ".join(exprs) if exprs else ""

    if query is not None:
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = db_utils.doc_obj_collection.search(
            data=[
                (
                    db_utils.embedder.embed([query])[0]
                    if type(query) is str
                    else query
                )
            ],
            anns_field=c.DOC_FLD_CONTENT_VECTOR,
            param=search_params,
            limit=limit,
            expr=expr or None,
            output_fields=output_fields,
        )
        hits = results[0] if results else []
        return [
            (
                hit.distance,
                DocumentObject(
                    **(
                        hit.entity["entity"]
                        | {
                            c.DOC_FLD_TYPE: idx_to_chunk_type(
                                hit.entity["entity"][c.DOC_FLD_TYPE]
                            )
                        }
                    )
                ),
            )
            for hit in hits
            if "entity" in hit.entity
        ]
    else:
        results = db_utils.doc_obj_collection.query(
            expr=expr, output_fields=output_fields, limit=limit, offset=skip
        )
        return [
            (
                None,
                DocumentObject(
                    **(
                        r
                        | {
                            c.DOC_FLD_TYPE: idx_to_chunk_type(
                                r[c.DOC_FLD_TYPE]
                            )
                        }
                    )
                ),
            )
            for r in results
        ]


# --- Omni functions ---


def o_search(
    query: Optional[str] = None,
    keywords: Optional[list[str]] = None,
    uploaded_before: Optional[int] = None,
    uploaded_after: Optional[int] = None,
    extra_fields: Optional[dict[str, Any]] = None,
    limit: int = 10,
) -> list[tuple[float | None, Union[Image, Span, DocumentObject]]]:
    query = query and db_utils.embedder.embed([query])[0]
    r1 = search_images(
        query=query,
        keywords=keywords,
        uploaded_before=uploaded_before,
        uploaded_after=uploaded_after,
        extra_fields=extra_fields,
        limit=limit,
    )
    r2 = search_spans(
        long_query=query,
        keywords=keywords,
        uploaded_before=uploaded_before,
        uploaded_after=uploaded_after,
        extra_fields=extra_fields,
        limit=limit,
    )
    r3 = search_document_objects(
        query=query,
        keywords=keywords,
        uploaded_before=uploaded_before,
        uploaded_after=uploaded_after,
        extra_fields=extra_fields,
        limit=limit,
    )
    results = r1 + r2 + r3
    if query is not None:
        results = sorted(results, key=lambda x: x[0], reverse=True)
    return results[:limit]
