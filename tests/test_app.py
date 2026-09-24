from app import parse_submission


def test_multipart_submission_extracts_question_and_supported_upload():
    boundary = "RabbitHoleBoundary"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"question\"\r\n\r\nWhy are gas prices rising?\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"sources\"; filename=\"private.md\"\r\nContent-Type: text/markdown\r\n\r\nPrivate supply evidence.\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    question, files = parse_submission(f"multipart/form-data; boundary={boundary}", body)

    assert question == "Why are gas prices rising?"
    assert files == [("private.md", b"Private supply evidence.")]
