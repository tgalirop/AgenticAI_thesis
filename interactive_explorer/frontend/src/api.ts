import type { ExplorerPayload } from "./types";

export async function fetchExplorer(): Promise<ExplorerPayload> {
  const response = await fetch("/api/explorer");
  if (!response.ok) {
    throw new Error(`Το explorer API επέστρεψε HTTP ${response.status}.`);
  }
  return response.json() as Promise<ExplorerPayload>;
}
