import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function timeAgo(iso: string): string {
  const then = new Date(iso).getTime();
  const now = Date.now();
  const secs = Math.max(1, Math.floor((now - then) / 1000));
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function pct(x: number, digits = 1): string {
  return `${(x * 100).toFixed(digits)}%`;
}

/** Format a UTC ISO timestamp (as stored in SQLite) in Eastern Time. */
export function formatEastern(iso: string | null | undefined): string {
  const parts = formatEasternParts(iso);
  if (!parts) return "";
  return `${parts.date}, ${parts.time} ${parts.tz}`;
}

/** Parse a UTC ISO timestamp and return the Eastern-time parts separately
 * (date, time, zone) so layouts can style each independently. */
export function formatEasternParts(
  iso: string | null | undefined,
): { date: string; time: string; tz: string } | null {
  if (!iso) return null;
  const normalized = iso.includes("T") ? iso : iso.replace(" ", "T");
  const asUtc =
    normalized.endsWith("Z") || /[+-]\d{2}:?\d{2}$/.test(normalized)
      ? normalized
      : normalized + "Z";
  const d = new Date(asUtc);
  if (isNaN(d.getTime())) return null;

  const date = d.toLocaleString("en-US", {
    timeZone: "America/New_York",
    month: "short",
    day: "numeric",
  });
  const time = d.toLocaleString("en-US", {
    timeZone: "America/New_York",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
  // Intl doesn't give us "EDT" vs "EST" as a clean field on all runtimes;
  // derive it by formatting with timeZoneName and stripping the rest.
  const withZone = d.toLocaleString("en-US", {
    timeZone: "America/New_York",
    hour: "numeric",
    hour12: true,
    timeZoneName: "short",
  });
  const tzMatch = withZone.match(/\b([ECMP][SD]T|UTC)\b/);
  const tz = tzMatch ? tzMatch[1] : "ET";

  return { date, time, tz };
}
