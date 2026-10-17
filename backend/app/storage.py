"""Vultr Object Storage (S3-compatible) access. The bucket stays private."""

import uuid
from datetime import datetime, timezone

import boto3
from botocore.config import Config
from fastapi.concurrency import run_in_threadpool

from .config import Settings


class ObjectStore:
    def __init__(self, settings: Settings):
        self.bucket = settings.s3_bucket
        self.ttl = settings.presigned_url_ttl_seconds
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url or None,
            region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key or None,
            aws_secret_access_key=settings.s3_secret_key or None,
            config=Config(signature_version="s3v4"),
        )

    @staticmethod
    def new_image_key(device_id: str) -> str:
        now = datetime.now(timezone.utc)
        safe_device = device_id.replace("@", "_").replace("|", "_")
        return f"images/{safe_device}/{now:%Y/%m/%d}/{uuid.uuid4().hex}.jpg"

    async def put_image(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        await run_in_threadpool(
            self.client.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return key

    def presigned_url(self, key: str) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.ttl,
        )

    def delete_keys(self, keys: list[str]) -> None:
        # S3 DeleteObjects accepts at most 1000 keys per call.
        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
            )
