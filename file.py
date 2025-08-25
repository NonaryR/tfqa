import logging
import os

import requests
import tenacity

from web import models
from web.data.models import CodetectorPayload
from web.data.models import CodetectorResponse


logger = logging.getLogger(__name__)

CODETECTOR_URL = os.getenv("CODETECTOR_URL")


def calculate_credits(tokens_processed: int) -> int:
    return int(tokens_processed // 100)


@tenacity.retry(
    wait=tenacity.wait_fixed(2),
    stop=tenacity.stop_after_delay(30),
    before_sleep=tenacity.before_sleep_log(logger, logging.CRITICAL),
)
def call_codetector(filename: str, file_content: str) -> CodetectorResponse:
    url = f"{CODETECTOR_URL}/detect_code"
    headers = {"Content-Type": "application/json"}
    data = {"filename": filename, "file_content": file_content}
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    return CodetectorResponse(**response.json())


def codetect_file(file_path, file_name) -> CodetectorPayload:
    with open(file_path, "rb") as file:
        file_content = file.read()

    extension = f".{file_name.split('.')[-1]}"

    try:
        language_map = models.SupportedProgrammingLanguages.objects.get(
            extension=extension
        )
        language = language_map.language
    except models.SupportedProgrammingLanguages.DoesNotExist:
        language = None

    if language is None or len(file_content) == 0:
        return CodetectorPayload(
            ai_detected=False,
            tokens_processed=0,
            annotations=[],
            spended_credits=0,
            programming_language=language,
            supported_language=False,
            file_extension=extension,
        )
    else:
        file_content = file_content.decode("utf-8")
        codetector_response = call_codetector(file_name, file_content)

        return CodetectorPayload(
            spended_credits=calculate_credits(codetector_response.tokens_processed),
            programming_language=language,
            supported_language=True,
            file_extension=extension,
            **codetector_response.model_dump(),
        )

