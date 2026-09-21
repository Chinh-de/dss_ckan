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

