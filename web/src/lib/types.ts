// Response shapes of the LastScene API (backend/app/routers). Keep in step with the backend.

export type BBox = [x: number, y: number, w: number, h: number];

/** A vision-pipeline detection drawn over a photo (normalised [x, y, w, h]). */
export interface Detection {
  label: string;
  confidence: number;
  bbox: BBox;
  maybe: boolean;
  event?: string;
}

export interface Item {
  name: string;
  location: string | null;
  last_seen: string;
  transcript: string;
  scene: string | null;
  image_url: string | null;
  detection: Detection | null;
  mentions: number;
  times_asked: number;
  previous: { location: string | null; timestamp: string }[];
}

export interface ItemsResponse {
  window_hours: number;
  items: Item[];
}

export interface HistoryEntry {
  id: string;
  device_id: string;
  timestamp: string;
  source: "manual" | "auto";
  transcript: string;
  scene: string | null;
  summary: string | null;
  detections: Detection[];
  image_url: string | null;
  items: { name: string; location: string | null }[];
}

export interface Question {
  timestamp: string;
  question: string;
  answer: string;
  item: string | null;
  answerable: boolean;
}

export interface Insights {
  window_hours: number;
  memories_logged: number;
  questions_asked: number;
  items_tracked: number;
  unanswered_count: number;
  most_asked: { name: string; count: number }[];
  most_logged: { name: string; count: number }[];
  unanswered: { timestamp: string; question: string; item: string | null }[];
  recent_questions: Question[];
  activity: { timestamp: string; kind: "memory" | "question" }[];
}

export interface SearchChunk {
  kind: "speech" | "scene" | "item" | "text";
  content: string;
  matched: boolean;
  vector_rank: number | null;
  keyword_rank: number | null;
}

export interface SearchResult {
  memory_id: string;
  timestamp: string;
  score: number;
  image_url: string | null;
  chunks: SearchChunk[];
}

export interface Room {
  id: string;
  name: string;
  polygon: [number, number][];
  putdowns: number;
  lost: number;
  items: string[];
}

export interface MapItem {
  name: string;
  room: string;
  room_name: string;
  location: string | null;
  last_seen: string;
  mentions: number;
  times_asked: number;
  position: [number, number];
}

export interface Trip {
  name: string;
  path: [number, number][];
  timestamps: number[];
}

export interface HouseMap {
  window_hours: number;
  start: string | null;
  span_hours: number;
  rooms: Room[];
  items: MapItem[];
  trips: Trip[];
}

export interface Place {
  id: string;
  label: string;
  name?: string | null;
  kind: "private" | "public";
  is_home: boolean;
  lat: number;
  lng: number;
  radius_m: number;
  visits: number;
}

export interface PlacesResponse {
  households: { id: string; name: string; places: Place[] }[];
  memories: { lat: number; lng: number; accuracy_m: number | null; place_id: string | null; items: string; transcript: string; timestamp: string }[];
}

export interface Household {
  id: string;
  name: string;
  share_public: boolean;
  role: "owner" | "member";
  members: { display_name: string; role: "owner" | "member"; is_you: boolean }[];
  devices: { client_id: string; label: string | null; last_seen: string | null }[];
  places: Place[];
}

export interface LostItem {
  name: string;
  left: number;
  asked: number;
}

export interface LostCell {
  lat: number;
  lng: number;
  left: number;
  asked: number;
  items: LostItem[];
}

export interface LostMap {
  min_households: number;
  cells: LostCell[];
  simulated?: boolean;
}
