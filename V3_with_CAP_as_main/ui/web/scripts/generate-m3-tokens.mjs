// Generates the Material 3 colour roles (light and dark) from one seed colour, using Google's
// Material Color Utilities (tonal-spot scheme, standard contrast), and prints them as CSS.
//
//   node --import ./scripts/register-loader.mjs scripts/generate-m3-tokens.mjs "#2a78d6"
//
// Paste the output into src/styles.css to re-theme the whole app from another seed.
import { Hct, SchemeTonalSpot, argbFromHex, hexFromArgb } from "@material/material-color-utilities";

const seed = process.argv[2] ?? "#2a78d6";
const roles = [
  "primary", "onPrimary", "primaryContainer", "onPrimaryContainer",
  "secondary", "onSecondary", "secondaryContainer", "onSecondaryContainer",
  "tertiary", "onTertiary", "tertiaryContainer", "onTertiaryContainer",
  "error", "onError", "errorContainer", "onErrorContainer",
  "surface", "surfaceDim", "surfaceBright", "onSurface", "onSurfaceVariant",
  "surfaceContainerLowest", "surfaceContainerLow", "surfaceContainer", "surfaceContainerHigh", "surfaceContainerHighest",
  "outline", "outlineVariant", "inverseSurface", "inverseOnSurface", "inversePrimary", "shadow", "scrim",
];
const kebab = (name) => name.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);

for (const dark of [false, true]) {
  const scheme = new SchemeTonalSpot(Hct.fromInt(argbFromHex(seed)), dark, 0);
  console.log(dark ? ':root[data-theme="dark"] {' : ":root {");
  for (const role of roles) console.log(`  --md-sys-color-${kebab(role)}: ${hexFromArgb(scheme[role])};`);
  console.log("}");
}
