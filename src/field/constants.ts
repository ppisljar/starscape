/* The one CSS-pixel unit module (DESIGN.md §5). The shader, the picker (stage 4) and the zoom
 * ladder (stage 7) must all agree about node size — if they disagree, the thing you click is
 * not the thing you see. These constants exist exactly once; three copies of a constant is
 * three chances to disagree (the predecessor's README says the same about NODE_SCALE /
 * NODE_MIN_PX / NODE_MAX_PX, exported from its field.js for the same reason).
 *
 * All *_PX values are CSS pixels, never drawing-buffer pixels. The devicePixelRatio is handled
 * exactly once, by the renderer (`renderer.setPixelRatio` in the consuming app); every pixel
 * number in the field's math stays in CSS px, so the ladder (which lays out DOM in CSS px) and
 * the shader can never be a device-pixel-ratio apart.
 *
 * Defaults are the predecessor's schema defaults (vault-site-3d src/config.js), which were
 * argued over on a real corpus; a knob per value arrives with the config schema at stage 8.
 */

/** World-radius multiplier applied to every node's size-channel radius. */
export const NODE_SCALE = 1.15;

/** A node's projected radius never falls below this many CSS px — a distant page stays a mote. */
export const NODE_MIN_PX = 2.2;

/** A FINITE upper px cap, used as a reference ceiling (the picker's hit-radius bound, the
 *  cap-slider's usable band). It is NOT the shipped shader default: the default is uncapped
 *  (NODE_MAX_PX_OFF) so a node you fly up to grows without bound and fills the frame. */
export const NODE_MAX_PX = 260;

/** The "no upper cap" sentinel for a node's projected-px clamp — the SHIPPED DEFAULT. A node's
 *  drawn size is bounded only BELOW (NODE_MIN_PX keeps a distant page a mote); it is free to
 *  grow without limit as the camera closes, so flying up to ANY node fills the view with no
 *  "pushing" wall (a finite cap holds screen size constant by shrinking the world radius as
 *  depth→0, which reads as the node backing away — exactly what we are removing). A finite cap
 *  can still be re-imposed via `setNodeStyle({ maxPx })` / the cap slider. 1e7 CSS px is past
 *  any real viewport, so the optional `min(px, uMaxPx)` against it never binds. */
export const NODE_MAX_PX_OFF = 1e7;

/** Under this projected radius (CSS px) the glyph draws the cheap flat branch: one flat colour,
 *  no halo, no smoothsteps, no form test. At that size the shape channel is not readable and
 *  pretending otherwise just costs fill (SCALE.md: most of the field at 10⁶). */
export const FLAT_PX = 2;

/** Rank-normalised size channel: world radii swept from MIN to MAX by percentile^SIZE_CURVE. */
export const SIZE_MIN_WORLD = 1.8;
export const SIZE_MAX_WORLD = 22;

/** Exponent on the percentile — what separates "a field of equal dots" from "dust with a few
 *  bright cores in it". 0.5 is the sqrt curve degree always used. */
export const SIZE_CURVE = 0.5;

/** Uniform sizing ('uniform' channel): one radius for every node. */
export const SIZE_UNIFORM_WORLD = (SIZE_MIN_WORLD + SIZE_MAX_WORLD) * 0.33;

/** Solid part of the glyph body, as a fraction of the sprite radius. */
export const BODY_R = 0.62;

/** Halo reach beyond the body, in CSS px — edge softness is a property of the SCREEN, not of
 *  the object, so it belongs in pixels (predecessor's hard-won rule). */
export const HALO_PX = 2.5;

/** Halo strength relative to the body. */
export const HALO_GAIN = 0.4;

/** How much the small bright centre burns toward white. */
export const CORE_HOT = 0.6;

/** Fake-sphere shading strength (0..1). 0 = a flat disc (the shipped default — the dark looks
 *  are unchanged); 1 = the body is shaded as a lit billboard "ball" (a hemisphere normal, a
 *  Lambert term lit from the upper-left, a soft specular). No geometry — the two-draw-call path
 *  is untouched; the shade is faked per-fragment. The `paper` look turns it on. */
export const SPHERE = 0;

/** Perspective bend on projected size. 1 = the true projection; below 1 distant pages stay
 *  larger than they should (field reads flat); above 1 depth is exaggerated. */
export const FALLOFF = 1.0;

/** The falloff pivot distance, as a fraction of the field diagonal — turning the falloff knob
 *  must not also resize what is right in front of you. (The predecessor used spacing×4 with a
 *  100-unit spacing on a ~4000-unit field; this is the same point, scale-free.) */
export const D_REF_FRACTION = 0.1;

/** Haze start/end, × field diagonal. Fog is atmosphere, not a cull. */
export const FOG_NEAR = 0.75;
export const FOG_FAR = 2.1;

/** Resting link alpha. */
export const LINK_BASE_ALPHA = 0.075;

/** Long links fade: alpha eases toward LINK_LEN_FLOOR over LINK_LEN_SOFT × diagonal of world
 *  length — a chord across the whole field carries almost no local information and costs the
 *  most fill. */
export const LINK_LEN_SOFT = 0.22;
export const LINK_LEN_FLOOR = 0.1;

/** The dark ground the default theme assumes (one of DESIGN.md §5's theme basics). */
export const BG_DARK = 0x05070d;

/** The light ground (the config schema's `background: 'light'`; paper's "ink on paper").
 *  Shipped beside BG_DARK so the named light-ground looks mean the same paper in every
 *  host (§7) — a host-invented ground made the same preset render differently per app. */
export const BG_LIGHT = 0xf2efe7;

/** A TRUE black ground (the config schema's `background: 'black'`). BG_DARK's #05070d is a near-
 *  black with a trace of blue that the additive nebula look reveals as a lifted, greyish negative
 *  space; the #4/#5 reference is pure #000 between the clusters, which only a real black ground and
 *  a 0→0 roll-off (reinhard) give. The nebula preset selects this. */
export const BG_BLACK = 0x000000;

/** Default vertical field of view, degrees. */
export const DEFAULT_FOV = 55;

/** How many distinguishable glyph families the fragment shader draws (disc, square, diamond,
 *  ring, cross, disc-in-cross, bar). An eighth nobody can name is worse than wrapping. */
export const GLYPH_FAMILIES = 7;
