### Gobbler

#### Glossary

- `Span:` A segment of a video.
- `Frame:` Videos are images shown one after another quickly, to simulate motion, a frame is one of these images.
- `Collection:` Milvus equivalent of an SQL table.

---

#### Collections schemas

#### spans

- `uri:` VARCHAR(1024) - unique resource identifier for the video (source)
- `id:` INT64 - primary key, auto-incremented
- `mime_type:` VARCHAR(128) - media type of the resource
- `size:` INT64 - size of the resource in bytes
- `uploaded_by:` VARCHAR(256) - user who uploaded the resource
- `uploaded_at:` INT64 - upload timestamp (epoch)
- `version:` FLOAT - version number of the resource
- `hash:` VARCHAR(64) - hash of the resource for integrity
- `start:` FLOAT - start time of the span
- `end:` FLOAT - end time of the span
- `short_description:` VARCHAR(65535) - brief description of the span
- `long_description:` VARCHAR(65535) - detailed description of the span
- `short_description_vector:` FLOAT_VECTOR(3072) - embedding of the short description
- `long_description_vector:` FLOAT_VECTOR(3072) - embedding of the long description
- `keywords:` JSON - keywords/tags for the span

#### images

- `id:` INT64 - primary key, auto-incremented
- `uri:` VARCHAR(1024) - unique resource identifier for the image
- `mime_type:` VARCHAR(128) - media type of the image
- `size:` INT64 - size of the image in bytes
- `uploaded_by:` VARCHAR(256) - user who uploaded the image
- `uploaded_at:` INT64 - upload timestamp (epoch)
- `version:` FLOAT - version number of the image
- `hash:` VARCHAR(64) - hash of the image for integrity
- `shape:` VARCHAR(16) - shape of the image (e.g., "1920x1080")
- `scene:` INT8 (nullable) - scene classification label
- `description:` VARCHAR(65535) - description of the image
- `description_vector:` FLOAT_VECTOR(3072) - embedding of the description
- `span_id:` INT64 (nullable) - reference to related span
- `keywords:` JSON - keywords/tags for the image

#### document_objects

- `uri:` VARCHAR(1024) - unique resource identifier for the source document
- `id:` INT64 - primary key, auto-incremented
- `mime_type:` VARCHAR(128) - media type of the document object
- `size:` INT64 - size of the object in bytes
- `uploaded_by:` VARCHAR(256) - user who uploaded the object
- `uploaded_at:` INT64 - upload timestamp (epoch)
- `version:` FLOAT - version number of the object
- `hash:` VARCHAR(64) - hash of the object for integrity
- `page:` INT8 - page number in the document
- `position:` JSON - position of the object on the page
- `type:` INT8 - type/classification of the object
- `content:` VARCHAR(65535) - textual content of the object
- `content_vector:` FLOAT_VECTOR(3072) - embedding of the content
- `keywords:` JSON - keywords/tags for

---

#### Getting started
- Ensure [ffmpeg](https://ffmpeg.org/download.html) is installed (skip if you don't need video processing).
- Build and run `Dockerfile.converter`, used to convert documents to PDF for easier processing.
  ```bash
  cd <project_directory>
  docker build -f Dockerfile.converter -t <image_name> .
  docker run -d -p 8000:8000 <image_name>
  ```
- Install Gobbler.
  ```bash
  pip install "git+https://dev.azure.com/Zifo/AIdeate%20and%20AIterate/_git/Multi-Modal%20Data%20Ingestion%20Pipeline"
  ```
  > NOTE: the code may not have been merged to main branch, try "git+https://dev.azure.com/Zifo/AIdeate%20and%20AIterate/_git/Multi-Modal%20Data%20Ingestion%20Pipeline@dev/nitesh" in case installation fails

- Ensure all environment variables are set.

- Initialize Gobbler.
  ```bash
  from gobbler import init
  init()
  ```
- Search away.
  ```bash
  from gobbler import search_images
  images = search_images(query="cat on a mat")

---

#### Core functions exposed to user

There are `8` core functions to insert and search for information.

---

#### `1. from gobbler import init`

```python
def init():
    ...
```

`init()` initializes the Milvus connection and other utilities, it must be called for before both insertions and searches.

---

#### Common function parameters

Some parameters are common to all core functions, they are explained here to reduce redundant lines of text.

> _common parameters for ingest\_\* functions_

- `uri:` The path to the file being ingested, can be a local path or an http/s link.
- `uploaded_by:` ID of the user uploading the input file, default is `system`.
- `version:` Version of the file being uploaded, default is `int(<modification_timestamp_of_the_file>)`.
- `processor:` Images, videos and documents have specific processor classes, this parameters helps the user pass a custom processor instance to be used configured to their needs, a processor instance with generic config is used otherwise.

  - for images, use `from gobbler.processors.image.core import ImageProcessor`
  - for videos, use `from gobbler.processors.video.core import VideoProcessor`
  - for images, use `from gobbler.processors.docs.core import DocumentProcessor`

- `extra_fields:` Dynamic fields.
- `download_headers:` HTTP headers used for downloading the URI's content if it is an http/s link.
- `throw_if_duplicate:` If set to `True`, this parameters makes the function throw a `ValueError` when the file being processed already exists in the database, the file's SHA-256 hash is used to check for duplicacy.

> _common parameters for search\_\* functions_

- `query:` Natural language query.
- `uploaded_by:` ID of the user who uploaded the file.
- `keywords:` List of keywords to check in the uploaded files.
- `mime_type:` Filters by the given mimetype, e.g. `image/png`, `video/mp4`.
- `uploaded_before:` Returns files uploaded before the given timestamp, in seconds.
- `uploaded_after:` Returns files uploaded after the given timestamp, in seconds.
- `extra_fields:` Dynamic fields.
- `limit:` Expects an integer `n`, limits the number of returned matches to `n`.
- `skip:` Expects an integer `n`, skips the first `n` matches.
  > Note: skipping is not supported when the search criteria includes vector similarity.

---

#### `2. from gobbler import ingest_image`

```python
def ingest_image(
    uri: str,
    uploaded_by: str = "system",
    version: Optional[float] = None,
    scene: Optional[SceneType] = None,
    span_id: Optional[int] = None,
    processor: Optional[ImageProcessor] = None,
    download_headers: Optional[dict[str, str]] = None,
    throw_if_duplicate: bool = True,
) -> Image:
```

- `scene:` Category of the content in the image, this helps choose a specific prompt to extract the content, e.g. if it's a diagram, the prompt empazises prompt specific instructions. Should be an attribute of the `gobbler.processors.image.utils.SceneType` enum.
- `span_id:` Images may be linked to a span, this links the image being processed to a span. e.g. the image is a screenshot from a video. This would rarely be used directly.

#### `3. from gobbler import search_images`

```python
def search_images(
    query: Optional[str] = None,
    mime_type: Optional[str] = None,
    uploaded_by: Optional[str] = None,
    scene: Optional[SceneType] = None,
    keywords: Optional[list[str]] = None,
    span_id: Optional[int] = None,
    uploaded_before: Optional[float] = None,
    uploaded_after: Optional[float] = None,
    limit: int = 10,
    skip: int = 0,
) -> list[tuple[float | None, Image]]:
```

Returns a list of tuples, the first element of the tuples is cosine score when `query` is passed, `None` otherwise. The list is sorted in descending order of scores.

Parameters:

- `scene:` Filter by image category, must be an attribute of `gobbler.processors.image.utils.SceneType` enum.
- `span_id:` This returns images linked to a span.

---

#### `4. from gobbler import ingest_video`

```python
def ingest_video(
    uri: str,
    uploaded_by: str = "system",
    version: Optional[float] = None,
    processor: Optional[VideoProcessor] = None,
    download_headers: Optional[dict[str, str]] = None,
    throw_if_duplicate: bool = True,
) -> list[Span]:
```

> Look at common parameters of ingest\_\* above.

#### `5. from gobbler import search_spans`

```python
def search_spans(
    short_query: Optional[str] = None,
    long_query: Optional[str] = None,
    mime_type: Optional[str] = None,
    keywords: Optional[list[str]] = None,
    uploaded_before: Optional[int] = None,
    uploaded_after: Optional[int] = None,
    limit: int = 10,
    skip: int = 0,
) -> list[tuple[float | None, Span]]:
```

Returns a list of tuples, the first element of the tuples is cosine score when `short_query` or `long_query` is passed, `None` otherwise. The list is sorted in descending order of scores.

Parameters:

- `short_query:` This matches against the field `short_description` of the spans collection.
- `long_query:` This matches against the field `long_description` of the spans collection.

---

### `6. from gobbler import ingest_document`

```python
def ingest_document(
    uri: str,
    uploaded_by: str = "system",
    version: Optional[float] = None,
    processor: Optional[DocumentProcessor] = None,
    download_headers: Optional[dict[str, str]] = None,
    throw_if_duplicate: bool = True,
) -> list[DocumentObject]:
```

> Look at common parameters of ingest\_\* above.

### `7. from gobbler import search_document_objects`

```python
def search_document_objects(
    query: Optional[str] = None,
    mime_type: Optional[str] = None,
    page: Optional[int] = None,
    type: Optional[ChunkType] = None,
    keywords: Optional[list[str]] = None,
    uploaded_before: Optional[int] = None,
    uploaded_after: Optional[int] = None,
    limit: int = 10,
    skip: int = 0,
) -> list[tuple[float | None, DocumentObject]]:
```

Returns a list of tuples, the first element of the tuples is cosine score when `query` is passed, `None` otherwise. The list is sorted in descending order of scores.

Parameters:

- `page:` Index of a page of documents, matches against the `page` field of `document_objects` collection, starts from 0.
- `type:` Type of object from the document, can be `text`, `table`, `figure` or `marginalia`, must be an attribute of the `agentic_doc.common.ChunkType` enum.

#### `8. from gobbler import o_search`
```python
def o_search(
    query: Optional[str] = None,
    keywords: Optional[list[str]] = None,
    uploaded_before: Optional[int] = None,
    uploaded_after: Optional[int] = None,
    extra_fields: Optional[dict[str, Any]] = None,
    limit: int = 10,
) -> list[tuple[float | None, Union[Image, Span, DocumentObject]]]:
```
> Look at common parameters of search\_\* above.

Returns a list of tuples where the first element of the tuple is a cosine score and the second may be an `Image`, `Span` or `DocumentObject`.

---

#### Data models
`Image`, `Span` and `DocumentObject` have the same fields as their corresponding collections schemas, except for any vector fields. Data models also expose a `to_json()` method to get the data as a JSON object.
