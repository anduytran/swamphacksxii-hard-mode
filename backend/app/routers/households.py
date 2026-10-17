"""Household setup: who shares a wearer's memories, which devices, which places.

Typical flow: a caregiver sets LastScene up for someone else.
  1. POST /households               create "Mom's household" (the caller becomes owner)
  2. POST /households/{id}/devices   claim the wearable with its Client ID + pairing code
  3. POST /households/{id}/invites   one-time code for other family members
  4. POST /invites/accept            family member joins with the code
  5. PATCH /households/{id}/places/{place}   name auto-found places, mark public ones
  6. PATCH /households/{id}          opt in to the anonymous public map
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..auth import Principal, require_caregiver
from ..db import MemoryDB
from ..deps import get_db
from ..geo import home_place_id, place_label, place_unplaced

router = APIRouter(prefix="/api/v1", tags=["households"])


class NewHousehold(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    display_name: str | None = Field(None, max_length=80)


class ClaimDevice(BaseModel):
    client_id: str = Field(..., min_length=4, max_length=128, description="The wearable's Auth0 Client ID")
    pairing_code: str = Field(..., min_length=4, max_length=32)
    label: str | None = Field(None, max_length=80)


class AcceptInvite(BaseModel):
    code: str = Field(..., min_length=4, max_length=32)
    display_name: str | None = Field(None, max_length=80)


class HouseholdUpdate(BaseModel):
    share_public: bool


class PlaceUpdate(BaseModel):
    name: str | None = Field(None, max_length=80)
    kind: str | None = Field(None, pattern="^(private|public)$")


def device_id_for(client_id: str) -> str:
    client_id = client_id.strip()
    return client_id if "@" in client_id else f"{client_id}@clients"


async def _require_member(db: MemoryDB, household_id: str, user: Principal, owner: bool = False) -> str:
    role = await db.member_role(household_id, user.sub)
    if role is None or (owner and role != "owner"):
        raise HTTPException(status.HTTP_404_NOT_FOUND if role is None else status.HTTP_403_FORBIDDEN,
                            "Household not found" if role is None else "Only the household owner can do that")
    return role


async def _places_view(db: MemoryDB, household_id: str) -> list[dict]:
    places = await db.places_for(household_id)
    home = home_place_id(places)
    return [
        {
            "id": p.id,
            "label": place_label(p, home, i),
            "name": p.name,
            "kind": p.kind,
            "is_home": p.id == home,
            "lat": p.lat,
            "lng": p.lng,
            "radius_m": p.radius_m,
            "visits": p.visits,
        }
        for i, p in enumerate(places, 1)
    ]


@router.get("/households/me")
async def my_households(user: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db)):
    result = []
    for h in await db.households_for(user.sub):
        detail = await db.household_detail(h.id)
        result.append(
            {
                "id": h.id,
                "name": h.name,
                "share_public": h.share_public,
                "role": h.role,
                "members": [
                    {"display_name": m["display_name"] or "Member", "role": m["role"], "is_you": m["user_sub"] == user.sub}
                    for m in detail["members"]
                ],
                "devices": [
                    {
                        "client_id": d["device_id"].removesuffix("@clients"),
                        "label": d["label"],
                        "last_seen": d["last_seen"].isoformat() if d["last_seen"] else None,
                    }
                    for d in detail["devices"]
                ],
                "places": await _places_view(db, h.id),
            }
        )
    return {"households": result}


@router.post("/households", status_code=status.HTTP_201_CREATED)
async def create_household(body: NewHousehold, user: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db)):
    return {"id": await db.create_household(user.sub, body.name.strip(), body.display_name)}


@router.post("/households/{household_id}/devices")
async def claim_device(
    household_id: str, body: ClaimDevice, user: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db)
):
    await _require_member(db, household_id, user, owner=True)
    device_id = device_id_for(body.client_id)
    result = await db.claim_device(household_id, device_id, body.pairing_code, body.label)
    if result == "bad_code":
        # Same answer for "never seen" and "wrong code", so codes can't be probed.
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "That device hasn't connected with this pairing code. Turn it on, make sure it's online, and check the code.")
    if result == "taken":
        raise HTTPException(status.HTTP_409_CONFLICT, "That device already belongs to another household.")
    placed = await place_unplaced(db, household_id, [device_id])
    return {"device_id": device_id, "memories_placed": placed}


@router.post("/households/{household_id}/invites")
async def create_invite(household_id: str, user: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db)):
    await _require_member(db, household_id, user, owner=True)
    return {"code": await db.create_invite(household_id, user.sub), "expires_in_hours": 72}


@router.post("/invites/accept")
async def accept_invite(body: AcceptInvite, user: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db)):
    household_id = await db.accept_invite(body.code, user.sub, body.display_name)
    if household_id is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "That invite code is invalid, used or expired.")
    return {"household_id": household_id}


@router.patch("/households/{household_id}")
async def update_household(
    household_id: str, body: HouseholdUpdate, user: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db)
):
    await _require_member(db, household_id, user, owner=True)
    await db.set_share_public(household_id, body.share_public)
    return {"share_public": body.share_public}


@router.patch("/households/{household_id}/places/{place_id}")
async def update_place(
    household_id: str, place_id: str, body: PlaceUpdate,
    user: Principal = Depends(require_caregiver), db: MemoryDB = Depends(get_db),
):
    await _require_member(db, household_id, user)
    name = body.name.strip() if body.name is not None else None
    if not await db.update_place(household_id, place_id, name or None, body.kind):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Place not found")
    return {"places": await _places_view(db, household_id)}
