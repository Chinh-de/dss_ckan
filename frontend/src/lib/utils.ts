import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatYear(year?: number | null): string {
  return year ? `${year}` : "N/A";
}

export function formatScore(score: number): string {
  return `${Math.round(score * 100)}%`;
}

// Curated high-aesthetic placeholders matching Taste-Skill (Cinematic / Editorial)
export const DOMAIN_FALLBACK_IMAGES: Record<string, string> = {
  movie: "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=600&auto=format&fit=crop&q=80",
  book: "https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=600&auto=format&fit=crop&q=80",
  music: "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=600&auto=format&fit=crop&q=80",
};

/**
 * Normalizes image URLs to prevent Mixed Content (HTTP -> HTTPS)
 * and guarantees a fallback image URL.
 */
export function getNormalizedImageUrl(url?: string | null, domain: string = "movie"): string {
  if (!url || typeof url !== "string" || !url.trim().startsWith("http")) {
    return DOMAIN_FALLBACK_IMAGES[domain] || DOMAIN_FALLBACK_IMAGES.movie;
  }
  let normalized = url.trim();
  // Ensure secure protocol for Amazon CDN and Last.FM CDN
  if (normalized.startsWith("http://")) {
    normalized = normalized.replace("http://", "https://");
  }
  return normalized;
}

export function getDomainFallbackImage(domain: string = "movie"): string {
  return DOMAIN_FALLBACK_IMAGES[domain] || DOMAIN_FALLBACK_IMAGES.movie;
}


