# --- Azure/OpenAI ---
LLM_FLD_ROLE = "role"
LLM_FLD_CONTENT = "content"
LLM_FLD_TYPE = "type"
LLM_FLD_TEXT = "text"
LLM_FLD_URL = "url"
LLM_FLD_DETAIL = "detail"
LLM_FLD_IMAGE_URL = "image_url"

LLM_ROLE_SYSTEM = "system"
LLM_ROLE_USER = "user"

LLM_CONTENT_TYPE_TEXT = "text"
LLM_CONTENT_TYPE_IMAGE_URL = "image_url"

# --- lib constants ---
USAGE_AOAI_EMBEDDING = "aoai_embedding"
USAGE_AOAI_TRANSCRIPTION = "aoai_transcription"
USAGE_AOAI_COMPLETION = "aoai_completion"
USAGE_AOAI_OCR = "aoai_ocr"
USAGE_LAI_OCR = "lai_ocr"

FLD_MODEL = "model"
FLD_USAGE_PROMPT = "prompt_tokens"
FLD_USAGE_COMPLETION = "completion_tokens"

# --- DB fields ---

DB_FLD_ID = "id"
DB_FLD_URI = "uri"
DB_FLD_MIME_TYPE = "mime_type"
DB_FLD_SIZE = "size"
DB_FLD_VERSION = "version"
DB_FLD_HASH = "hash"
DB_FLD_UPLOADED_AT = "uploaded_at"
DB_FLD_UPLOADED_BY = "uploaded_by"
DB_FLD_KEYWORDS = "keywords"

# --- Image collection fields ---
COLL_NAME_IMAGES = "images"
IMG_FLD_SHAPE = "shape"
IMG_FLD_SCENE = "scene"
IMG_FLD_DESCRIPTION = "description"
IMG_FLD_DESCRIPTION_VECTOR = "description_vector"
IMG_FLD_SPAN_ID = "span_id"

# --- Span collection fields ---
COLL_NAME_SPANS = "spans"
SPAN_FLD_START = "start"
SPAN_FLD_END = "end"
SPAN_FLD_SHORT_DESCRIPTION = "short_description"
SPAN_FLD_SHORT_DESCRIPTION_VECTOR = "short_description_vector"
SPAN_FLD_LONG_DESCRIPTION = "long_description"
SPAN_FLD_LONG_DESCRIPTION_VECTOR = "long_description_vector"

# -- Document collection fields ---
COLL_NAME_DOCUMENT_OBJECTS = "document_objects"
DOC_FLD_PAGE = "page"
DOC_FLD_POSITION = "position"
DOC_FLD_TYPE = "type"
DOC_FLD_CONTENT = "content"
DOC_FLD_CONTENT_VECTOR = "content_vector"
