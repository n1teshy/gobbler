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

import core.cred as cred
from core.embedders.azure import AzureEmbedder

spans_collection: Collection | None = None
images_collection: Collection | None = None
embedder: AzureEmbedder | None = None


def init():
    global spans_collection, images_collection, embedder

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

        spans_fields = [
            FieldSchema(name="URI", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="mime_type", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="size", dtype=DataType.INT64),
            FieldSchema(name="uploaded_by", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="uploaded_at", dtype=DataType.FLOAT),
            FieldSchema(name="version", dtype=DataType.FLOAT),
            FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="start", dtype=DataType.FLOAT),
            FieldSchema(name="end", dtype=DataType.FLOAT),
            FieldSchema(name="duration", dtype=DataType.FLOAT),
            FieldSchema(
                name="short_description", dtype=DataType.VARCHAR, max_length=1024
            ),
            FieldSchema(
                name="long_description", dtype=DataType.VARCHAR, max_length=8192
            ),
            FieldSchema(
                name="short_description_vector", dtype=DataType.FLOAT_VECTOR, dim=3072
            ),
            FieldSchema(
                name="long_description_vector", dtype=DataType.FLOAT_VECTOR, dim=3072
            ),
            FieldSchema(name="keywords", dtype=DataType.JSON),
        ]

        if not utility.has_collection("spans"):
            spans_schema = CollectionSchema(
                fields=spans_fields,
                description="Video time spans collection",
                enable_dynamic_field=True,
            )
            spans_collection = Collection(name="spans", schema=spans_schema)
        else:
            spans_collection = Collection("spans")

        span_emb_index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }

        try:
            spans_collection.create_index(
                field_name="URI", index_params={"index_type": "TRIE"}
            )
        except MilvusException:
            pass

        try:
            spans_collection.create_index(
                field_name="short_description_vector",
                index_params=span_emb_index_params,
            )
        except MilvusException:
            pass

        try:
            spans_collection.create_index(
                field_name="long_description_vector", index_params=span_emb_index_params
            )
        except MilvusException:
            pass

        try:
            spans_collection.create_index(
                field_name="URI", index_params={"index_type": "TRIE"}
            )
        except MilvusException:
            pass

        images_fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="URI", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="mime_type", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="size", dtype=DataType.INT64),
            FieldSchema(name="uploaded_by", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="uploaded_at", dtype=DataType.FLOAT),
            FieldSchema(name="version", dtype=DataType.FLOAT),
            FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="shape", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(
                name="scene", dtype=DataType.VARCHAR, max_length=128, nullable=True
            ),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(
                name="description_vector", dtype=DataType.FLOAT_VECTOR, dim=3072
            ),
            FieldSchema(name="span_id", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="keywords", dtype=DataType.JSON),
        ]

        if not utility.has_collection("images"):
            images_schema = CollectionSchema(
                fields=images_fields,
                description="Image frames collection",
                enable_dynamic_field=True,
            )
            images_collection = Collection(name="images", schema=images_schema)
        else:
            images_collection = Collection("images")

        images_index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }

        try:
            images_collection.create_index(
                field_name="description_vector", index_params=images_index_params
            )
        except MilvusException:
            pass

        try:
            images_collection.create_index(
                field_name="hash", index_params={"index_type": "TRIE"}
            )
        except MilvusException:
            pass

        try:
            images_collection.create_index(
                field_name="span_id", index_params={"index_type": "STL_SORT"}
            )
        except MilvusException:
            pass

        try:
            images_collection.create_index(
                field_name="URI", index_params={"index_type": "TRIE"}
            )
        except MilvusException:
            pass

        spans_collection.load()
        images_collection.load()
