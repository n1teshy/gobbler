import os

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    db,
    utility,
)

import gobbler.constants as c
import gobbler.cred as cred
from gobbler.embedders.azure import AzureEmbedder

spans_collection: Collection | None = None
images_collection: Collection | None = None
doc_obj_collection: Collection | None = None
embedder: AzureEmbedder | None = None


def init():
    global spans_collection, images_collection, doc_obj_collection, embedder

    if embedder is None:
        embedder = AzureEmbedder()

    if spans_collection is None or images_collection is None:
        mm_db = os.getenv("MM_DB", "mm_index")
        connections.connect(
            host=cred.MILVUS_HOST,
            port=cred.MILVUS_PORT,
            user=cred.MILVUS_USER,
            password=cred.MILVUS_PASSWORD,
        )

        if mm_db not in db.list_database():
            db.create_database(mm_db)

        db.using_database(mm_db)

        # --- spans collection ---

        spans_fields = [
            FieldSchema(
                name=c.DB_FLD_URI, dtype=DataType.VARCHAR, max_length=1024
            ),
            FieldSchema(
                name=c.DB_FLD_ID,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name=c.DB_FLD_MIME_TYPE, dtype=DataType.VARCHAR, max_length=128
            ),
            FieldSchema(name=c.DB_FLD_SIZE, dtype=DataType.INT64),
            FieldSchema(
                name=c.DB_FLD_UPLOADED_BY,
                dtype=DataType.VARCHAR,
                max_length=256,
            ),
            FieldSchema(name=c.DB_FLD_UPLOADED_AT, dtype=DataType.INT64),
            FieldSchema(name=c.DB_FLD_VERSION, dtype=DataType.FLOAT),
            FieldSchema(
                name=c.DB_FLD_HASH, dtype=DataType.VARCHAR, max_length=64
            ),
            FieldSchema(name=c.SPAN_FLD_START, dtype=DataType.FLOAT),
            FieldSchema(name=c.SPAN_FLD_END, dtype=DataType.FLOAT),
            FieldSchema(
                name=c.SPAN_FLD_SHORT_DESCRIPTION,
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name=c.SPAN_FLD_LONG_DESCRIPTION,
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name=c.SPAN_FLD_SHORT_DESCRIPTION_VECTOR,
                dtype=DataType.FLOAT_VECTOR,
                dim=3072,
            ),
            FieldSchema(
                name=c.SPAN_FLD_LONG_DESCRIPTION_VECTOR,
                dtype=DataType.FLOAT_VECTOR,
                dim=3072,
            ),
            FieldSchema(name=c.DB_FLD_KEYWORDS, dtype=DataType.JSON),
        ]

        if not utility.has_collection(c.COLL_NAME_SPANS):
            spans_schema = CollectionSchema(
                fields=spans_fields,
                description="Video time spans collection",
                enable_dynamic_field=True,
            )
            spans_collection = Collection(
                name=c.COLL_NAME_SPANS, schema=spans_schema
            )
        else:
            spans_collection = Collection(c.COLL_NAME_SPANS)

        span_emb_index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }

        try:
            spans_collection.create_index(
                field_name=c.DB_FLD_URI, index_params={"index_type": "TRIE"}
            )
        except MilvusException:
            pass

        try:
            spans_collection.create_index(
                field_name=c.SPAN_FLD_SHORT_DESCRIPTION_VECTOR,
                index_params=span_emb_index_params,
            )
        except MilvusException:
            pass

        try:
            spans_collection.create_index(
                field_name=c.SPAN_FLD_LONG_DESCRIPTION_VECTOR,
                index_params=span_emb_index_params,
            )
        except MilvusException:
            pass

        try:
            spans_collection.create_index(
                field_name=c.DB_FLD_URI, index_params={"index_type": "TRIE"}
            )
        except MilvusException:
            pass

        # --- images collection ---

        images_fields = [
            FieldSchema(
                name=c.DB_FLD_ID,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name=c.DB_FLD_URI, dtype=DataType.VARCHAR, max_length=1024
            ),
            FieldSchema(
                name=c.DB_FLD_MIME_TYPE, dtype=DataType.VARCHAR, max_length=128
            ),
            FieldSchema(name=c.DB_FLD_SIZE, dtype=DataType.INT64),
            FieldSchema(
                name=c.DB_FLD_UPLOADED_BY,
                dtype=DataType.VARCHAR,
                max_length=256,
            ),
            FieldSchema(name=c.DB_FLD_UPLOADED_AT, dtype=DataType.INT64),
            FieldSchema(name=c.DB_FLD_VERSION, dtype=DataType.FLOAT),
            FieldSchema(
                name=c.DB_FLD_HASH, dtype=DataType.VARCHAR, max_length=64
            ),
            FieldSchema(
                name=c.IMG_FLD_SHAPE, dtype=DataType.VARCHAR, max_length=16
            ),
            FieldSchema(
                name=c.IMG_FLD_SCENE, dtype=DataType.INT8, nullable=True
            ),
            FieldSchema(
                name=c.IMG_FLD_DESCRIPTION,
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name=c.IMG_FLD_DESCRIPTION_VECTOR,
                dtype=DataType.FLOAT_VECTOR,
                dim=3072,
            ),
            FieldSchema(
                name=c.IMG_FLD_SPAN_ID, dtype=DataType.INT64, nullable=True
            ),
            FieldSchema(name=c.DB_FLD_KEYWORDS, dtype=DataType.JSON),
        ]

        if not utility.has_collection(c.COLL_NAME_IMAGES):
            images_schema = CollectionSchema(
                fields=images_fields,
                description="Image frames collection",
                enable_dynamic_field=True,
            )
            images_collection = Collection(
                name=c.COLL_NAME_IMAGES, schema=images_schema
            )
        else:
            images_collection = Collection(c.COLL_NAME_IMAGES)

        try:
            images_collection.create_index(
                field_name=c.DB_FLD_URI, index_params={"index_type": "TRIE"}
            )
        except MilvusException:
            pass

        try:
            images_collection.create_index(
                field_name=c.IMG_FLD_SCENE,
                index_params={"index_type": "STL_SORT"},
            )
        except MilvusException:
            pass

        try:
            images_collection.create_index(
                field_name=c.IMG_FLD_DESCRIPTION_VECTOR,
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
        except MilvusException:
            pass

        try:
            images_collection.create_index(
                field_name=c.DB_FLD_HASH, index_params={"index_type": "TRIE"}
            )
        except MilvusException:
            pass

        try:
            images_collection.create_index(
                field_name=c.IMG_FLD_SPAN_ID,
                index_params={"index_type": "STL_SORT"},
            )
        except MilvusException:
            pass

        # --- document objects collection ---

        doc_obj_fields = [
            FieldSchema(
                name=c.DB_FLD_URI, dtype=DataType.VARCHAR, max_length=1024
            ),
            FieldSchema(
                name=c.DB_FLD_ID,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name=c.DB_FLD_MIME_TYPE, dtype=DataType.VARCHAR, max_length=128
            ),
            FieldSchema(name=c.DB_FLD_SIZE, dtype=DataType.INT64),
            FieldSchema(
                name=c.DB_FLD_UPLOADED_BY,
                dtype=DataType.VARCHAR,
                max_length=256,
            ),
            FieldSchema(name=c.DB_FLD_UPLOADED_AT, dtype=DataType.INT64),
            FieldSchema(name=c.DB_FLD_VERSION, dtype=DataType.FLOAT),
            FieldSchema(
                name=c.DB_FLD_HASH, dtype=DataType.VARCHAR, max_length=64
            ),
            FieldSchema(name=c.DOC_FLD_PAGE, dtype=DataType.INT8),
            FieldSchema(name=c.DOC_FLD_POSITION, dtype=DataType.JSON),
            FieldSchema(name=c.DOC_FLD_TYPE, dtype=DataType.INT8),
            FieldSchema(
                name=c.DOC_FLD_CONTENT,
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name=c.DOC_FLD_CONTENT_VECTOR,
                dtype=DataType.FLOAT_VECTOR,
                dim=3072,
            ),
            FieldSchema(name=c.DB_FLD_KEYWORDS, dtype=DataType.JSON),
        ]

        if not utility.has_collection(c.COLL_NAME_DOCUMENT_OBJECTS):
            doc_obj_schema = CollectionSchema(
                fields=doc_obj_fields,
                description="Document object collection",
                enable_dynamic_field=True,
            )
            doc_obj_collection = Collection(
                name=c.COLL_NAME_DOCUMENT_OBJECTS, schema=doc_obj_schema
            )
        else:
            doc_obj_collection = Collection(c.COLL_NAME_DOCUMENT_OBJECTS)

        try:
            doc_obj_collection.create_index(
                field_name=c.DB_FLD_URI, index_params={"index_type": "TRIE"}
            )
        except MilvusException:
            pass

        try:
            doc_obj_collection.create_index(
                field_name=c.DOC_FLD_TYPE,
                index_params={"index_type": "STL_SORT"},
            )
        except MilvusException:
            pass

        try:
            doc_obj_collection.create_index(
                field_name=c.DOC_FLD_CONTENT_VECTOR,
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
        except MilvusException:
            pass

        spans_collection.load()
        images_collection.load()
        doc_obj_collection.load()
