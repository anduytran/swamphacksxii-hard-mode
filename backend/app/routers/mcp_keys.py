"""Household MCP keys: read-only access to /mcp for scripts (e.g. a Gemini call), no login.

    GET    /api/v1/households/{id}/mcp-keys           active keys (labels and hints only)
    POST   /api/v1/households/{id}/mcp-keys           make one; the key is in this answer only
    DELETE /api/v1/households/{id}/mcp-keys/{key_id}  revoke it

Only the household owner makes and revokes keys. See docs/MCP.md.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..auth import Principal, require_caregiver
from ..db import MemoryDB
from ..deps import get_db
from .households import _require_member

MAX_KEYS_PER_HOUSEHOLD = 5

router = APIRouter(prefix="/api/v1/households", tags=["mcp"])


class NewKey(BaseModel):
    label: str = Field(..., min_length=1, max_length=80)


def _key_json(k: dict) -> dict:
    return {
        "id": k["id"],
        "label": k["label"],
        "hint": k["hint"] + "…",
        "created_at": k["created_at"].isoformat(),
        "last_used_at": k["last_used_at"].isoformat() if k["last_used_at"] else None,
    }


@router.get("/{household_id}/mcp-keys")
async def list_keys(household_id: str, user: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db)):
    await _require_member(db, household_id, user)
    return {"keys": [_key_json(k) for k in await db.mcp_keys(household_id)]}


@router.post("/{household_id}/mcp-keys", status_code=status.HTTP_201_CREATED)
async def create_key(
    household_id: str, body: NewKey, user: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db)
):
    await _require_member(db, household_id, user, owner=True)
    if len(await db.mcp_keys(household_id)) >= MAX_KEYS_PER_HOUSEHOLD:
        raise HTTPException(status.HTTP_409_CONFLICT, f"A household can have up to {MAX_KEYS_PER_HOUSEHOLD} MCP keys; revoke one first")
    key_id, key = await db.create_mcp_key(household_id, user.sub, body.label.strip())
    return {"id": key_id, "key": key, "note": "Shown once. Store it like a password; revoke it here if it leaks."}


@router.delete("/{household_id}/mcp-keys/{key_id}")
async def revoke_key(
    household_id: str, key_id: str, user: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db)
):
    await _require_member(db, household_id, user, owner=True)
    try:
        uuid.UUID(key_id)  # not a UUID: no such key (the ::uuid cast would otherwise be a 500)
    except ValueError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No active key with that id") from None
    if not await db.revoke_mcp_key(household_id, key_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No active key with that id")
    return {"revoked": key_id}
