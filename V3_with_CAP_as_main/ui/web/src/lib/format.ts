export const pct = (value: number | null | undefined, digits = 1): string =>
  value === null || value === undefined || Number.isNaN(value) ? "n/a" : `${(value * 100).toFixed(digits)}%`;

export const num = (value: number | null | undefined, digits = 2): string =>
  value === null || value === undefined || Number.isNaN(value) ? "n/a" : value.toFixed(digits);

export const int = (value: number | null | undefined): string =>
  value === null || value === undefined || Number.isNaN(value) ? "n/a" : Math.round(value).toLocaleString("en-US");

export const compact = (value: number): string => {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 0 : 1)}M`;
  if (value >= 10_000) return `${(value / 1000).toFixed(0)}K`;
  return value.toLocaleString("en-US");
};

export const duration = (seconds: number | null | undefined): string => {
  if (!seconds) return "n/a";
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return h ? `${h} h ${m} min` : `${m} min`;
};

/** "573-575, 580" style listing for volume numbers. */
export const ranges = (values: string[]): string => {
  const numbers = values.map(Number).filter((n) => !Number.isNaN(n)).sort((a, b) => a - b);
  const out: string[] = [];
  let start = numbers[0];
  let prev = numbers[0];
  for (let i = 1; i <= numbers.length; i += 1) {
    const current = numbers[i];
    if (current === prev + 1) {
      prev = current;
      continue;
    }
    if (start !== undefined) out.push(start === prev ? `${start}` : `${start}–${prev}`);
    start = current;
    prev = current;
  }
  return out.join(", ");
};
