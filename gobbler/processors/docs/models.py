import json
import re
from collections import namedtuple
from html import unescape
from typing import Optional

from gobbler.processors.interfaces import BaseFile, DBEntity

Position = namedtuple("Position", ["x1", "y1", "x2", "y2"])


class DocumentObject(DBEntity, BaseFile):
    page: int
    position: Position
    type: Optional[str]
    content: str
    keywords: list[str] = []

    class Config:
        extra = "allow"

    @property
    def table(self) -> Optional[list[list]]:
        if self.type != "table":
            return None

        table_match = re.search(
            r"<table.*?>.*?</table>", self.content, re.DOTALL | re.IGNORECASE
        )
        if not table_match:
            return []
        table_html = re.sub(r"\s+", " ", table_match.group(0))
        rows = re.findall(
            r"<tr.*?>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE
        )
        parsed_table = []
        for row_html in rows:
            cells = re.findall(
                r"<t[dh].*?>(.*?)</t[dh]>", row_html, re.DOTALL | re.IGNORECASE
            )
            cell_texts = [
                unescape(re.sub(r"<.*?>", "", cell)).strip() for cell in cells
            ]
            parsed_table.append(cell_texts)
        max_cols = max(len(row) for row in parsed_table)
        return [
            (row + ([None] * (max_cols - len(row)))) for row in parsed_table
        ]

    def to_json(self) -> dict:
        return json.loads(super().model_dump_json())
