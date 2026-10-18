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

/** A public place named by Grounding with Google Maps; always shown with its sources (MapsPlace). */
export interface PlaceName {
  name: string;
  sources: { title: string; url: string }[];
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
  place_name?: PlaceName | null;
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
  kind?: "room" | "hall"; // hall: a level's corridors, drawn under its rooms
  level?: number; // floor, 0 = ground (several only on multi-floor plans)
  polygon: [number, number][];
  putdowns: number;
  lost: number;
  items: string[];
}

export interface MapItem {
  name: string;
  room: string;
  room_name: string;
  room_source?: "photo" | "words"; // photo: the room the camera showed; words: matched from what was said
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
  plan?: { id: string; name: string }; // which floor plan the rooms come from
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
  memories: {
    lat: number;
    lng: number;
    accuracy_m: number | null;
    place_id: string | null;
    items: string;
    transcript: string;
    timestamp: string;
    place_name?: PlaceName | null;
  }[];
}

export interface Household {
  id: string;
  name: string;
  share_public: boolean;
  language: string; // BCP-47, the wearer's spoken answers (see lib/languages.ts)
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
  place_name?: PlaceName | null;
}

export interface LostMap {
  min_households: number;
  cells: LostCell[];
  simulated?: boolean;
}

/** A confirmed Teach photo. Absent photos have no box and stay out of ordinary YOLO training. */
export interface TrainingSample {
  id: string;
  annotation?: "present" | "absent" | "excluded" | "unreviewed"; // older API responses imply present
  bbox: [number, number, number, number] | null;
  width: number;
  height: number;
  image_url: string;
  created_at: string;
}

export interface TrainingItem {
  id: string;
  name: string;
  aliases: string[];
  created_at: string;
  samples: TrainingSample[];
}

export interface TrainingList {
  items: TrainingItem[];
  limits: { items: number; samples_per_item: number };
}

/** Hourly counts for the "when things go missing" grid; `at` is a UTC hour. */
export interface MissingTimes {
  days: number;
  since: string;
  hours: { at: string; asked: number; put_down: number }[];
}

/** GET /households/{id}/training/model: is the camera's model up to date with the Teach photos? */
export interface ModelStatus {
  dataset_version: string;
  photos: number;
  model: { version: string; trained_at: string | null; map50: number | null; base_model: string | null } | null;
  up_to_date: boolean;
}

/** One "Needs a look" card: a photo the detector is unsure of (detection) or that names the item with no box (memory). */
export interface ReviewCandidate {
  item_id: string;
  item_name: string;
  memory_id: string;
  kind: "detection" | "memory";
  image_url: string;
  captured_at: string;
  bbox: [number, number, number, number] | null;
  confidence: number | null;
  width: number | null;
  height: number | null;
}

/** GET /households/{id}/training/review */
export interface ReviewQueue {
  count: number;
  per_item: Record<string, number>;
  candidates: ReviewCandidate[];
  excluded?: ReviewCandidate[];
}

/** One voice command on the Live page (GET /api/v1/dashboard/live). */
export interface VoiceEvent {
  id: number;
  device_id: string;
  created_at: string;
  kind: string; // wake | voice | button: remember | button: ask
  outcome: string; // saved | answered | found | ignored | nothing to do | error
  heard?: string;
  words?: string; // what counts: the words after "Last Scene"
  question?: string;
  continued?: string; // "continue": the earlier question it follows on from
  intent?: string;
  item?: string;
  item_from?: string;
  decided_by?: "model" | "rules";
  command_error?: string;
  photos: number;
  best_photo?: number;
  in_view?: boolean;
  seen_where?: string;
  photo_error?: string;
  saved_photo?: number;
  memory_id?: string;
  image_url?: string;
  answer?: string;
  answered_by?: string;
  error?: string;
  steps: Record<string, number>; // step -> ms, in the order they ran
  served: Record<string, string>; // step -> provider/model
  total_ms: number;
  device?: {
    recorded_s?: number;
    ended?: string; // drop | silence | max
    photos?: number;
    photo_at_s?: number[];
    wake_score?: number;
    preroll_s?: number;
  };
  timeline?: Record<string, number>; // from the device: speech_end, photos_ready, sent, reply, played (s since wake)
  followup?: string; // running | done | failed: ...
  followup_ms?: number;
  photo_check_ms?: number; // picking the photo that shows the item, after the reply
}

/** The parts of /healthz the Live page shows. */
export interface Health {
  ok: boolean;
  missing_settings?: string[];
  model_usage?: Record<string, { used_today: number; limit_per_day: number }>;
  speech?: {
    provider: string;
    listening: string;
    monthly_character_cap?: number;
    characters_this_month?: number;
    monthly_audio_seconds_cap?: number;
    audio_seconds_this_month?: number;
  };
}
