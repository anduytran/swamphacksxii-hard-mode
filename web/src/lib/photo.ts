// Opening any photo people pick, including iPhone HEIC/HEIF. Safari reads HEIC itself; elsewhere the
// libheif decoder (heic-to, ~3 MB with its WebAssembly) loads on first use only, so JPEG uploads never pay for it.

export const PHOTO_ACCEPT = "image/*,.heic,.heif";

export function looksHeic(file: { name: string; type: string }): boolean {
  return /^image\/hei[cf](-sequence)?$/i.test(file.type) || /\.hei[cf]$/i.test(file.name);
}

/** The photo upright (EXIF/HEIF rotation applied), decoded natively when the browser can. */
export async function openPhoto(file: File): Promise<ImageBitmap> {
  try {
    return await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch (native) {
    const { heicTo, isHeic } = await import("heic-to");
    // some systems hand over HEIC with no type or a .jpg name, so check the bytes too
    if (!looksHeic(file) && !(await isHeic(file))) throw native;
    return heicTo({ blob: file, type: "bitmap", options: { imageOrientation: "from-image" } });
  }
}

/** onError for a server photo: try once more after a moment. A busy server can fail a few of the
 * many photo requests a page makes at once; a second try nearly always loads. */
export function retryPhoto(e: { currentTarget: HTMLImageElement }): void {
  const img = e.currentTarget;
  if (img.dataset.retried || img.src.startsWith("blob:")) return;
  img.dataset.retried = "1";
  const src = img.src;
  setTimeout(() => {
    if (img.src === src) img.src = `${src}${src.includes("?") ? "&" : "?"}retry=1`;
  }, 1500);
}
