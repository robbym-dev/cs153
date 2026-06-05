/** Mirror of bid_engine.pricing.normalize_unit — keeps client-side joins
 *  with reference data using the same canonical keys the backend uses. */
export function normalizeUnit(unit: string): string {
  const u = unit.trim().toUpperCase().replace(/\./g, "");
  if (["LF", "FT", "LIN FT", "LINEAR FT"].includes(u)) return "LF";
  if (["SF", "SQ FT", "SQFT"].includes(u)) return "SF";
  if (["EA", "EACH"].includes(u)) return "EA";
  if (["LS", "LUMP SUM"].includes(u)) return "LS";
  return u;
}
