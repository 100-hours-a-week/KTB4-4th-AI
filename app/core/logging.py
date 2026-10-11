from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str) -> None:
    """앱 로거(app.*)가 uvicorn 로그와 같은 출력으로 나가게 한다.

    uvicorn은 자기 로거만 설정해서, 따로 설정하지 않으면 앱 로그는 형식 없이 WARNING 이상만 나온다.
    이미 핸들러가 있으면(테스트, 다른 실행기) 건드리지 않는다.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level.upper(), format=LOG_FORMAT)
