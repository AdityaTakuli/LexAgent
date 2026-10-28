import { useId } from "react";

/** A classical Ionic column drawn in SVG: fluted shaft with cylindrical shading, a capital with
 *  scroll volutes, and a stepped base. Scales to any height; purely decorative. */
export function Column({ className = "", style }: { className?: string; style?: React.CSSProperties }) {
  const id = useId().replace(/:/g, "");
  const flutes = Array.from({ length: 9 }, (_, i) => 22 + i * 9.5);
  return (
    <svg viewBox="0 0 120 820" className={className} style={style} aria-hidden focusable="false" preserveAspectRatio="xMidYMax meet">
      <defs>
        {/* Light falls from the left: bright left third, darkening toward the right edge. */}
        <linearGradient id={`shaft-${id}`} x1="0" x2="1" y1="0" y2="0">
          <stop offset="0" stopColor="#bfb7a6" />
          <stop offset="0.18" stopColor="#f1ebdf" />
          <stop offset="0.42" stopColor="#e6dfd1" />
          <stop offset="0.78" stopColor="#c9c0ae" />
          <stop offset="1" stopColor="#a39a88" />
        </linearGradient>
        <linearGradient id={`stone-${id}`} x1="0" x2="1" y1="0" y2="0">
          <stop offset="0" stopColor="#cfc7b6" />
          <stop offset="0.3" stopColor="#f3eee4" />
          <stop offset="1" stopColor="#b3aa98" />
        </linearGradient>
        <linearGradient id={`shade-${id}`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#000" stopOpacity="0.18" />
          <stop offset="1" stopColor="#000" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* Capital: abacus, echinus and two volutes. */}
      <rect x="2" y="0" width="116" height="12" rx="1.5" fill={`url(#stone-${id})`} />
      <rect x="2" y="11" width="116" height="3" fill="#000" opacity="0.08" />
      <path d="M6 14 H114 Q116 30 104 34 H16 Q4 30 6 14 Z" fill={`url(#stone-${id})`} />
      <g fill="none" stroke="#9c937f" strokeWidth="1.6">
        <circle cx="14" cy="27" r="10" fill={`url(#stone-${id})`} />
        <path d="M14 27 m-6 0 a6 6 0 1 0 6 -6 a3.5 3.5 0 1 0 -3.5 3.5" />
        <circle cx="106" cy="27" r="10" fill={`url(#stone-${id})`} />
        <path d="M106 27 m6 0 a6 6 0 1 1 -6 -6 a3.5 3.5 0 1 1 3.5 3.5" />
      </g>
      <path d="M24 34 H96 L92 44 H28 Z" fill={`url(#stone-${id})`} />
      <rect x="20" y="44" width="80" height="6" fill="#000" opacity="0.07" />

      {/* Shaft with entasis (a slight swell) and flutes. */}
      <path d="M20 50 H100 L103 760 H17 Z" fill={`url(#shaft-${id})`} />
      {flutes.map((x) => (
        <g key={x}>
          <line x1={x} y1="56" x2={x + (x - 60) * 0.035} y2="754" stroke="#000" strokeOpacity="0.13" strokeWidth="2.2" strokeLinecap="round" />
          <line x1={x + 1.8} y1="56" x2={x + 1.8 + (x - 60) * 0.035} y2="754" stroke="#fff" strokeOpacity="0.28" strokeWidth="1" />
        </g>
      ))}
      <rect x="20" y="50" width="80" height="26" fill={`url(#shade-${id})`} />

      {/* Base: astragal, torus mouldings, plinth. */}
      <rect x="15" y="758" width="90" height="6" rx="3" fill={`url(#stone-${id})`} />
      <rect x="10" y="764" width="100" height="12" rx="6" fill={`url(#stone-${id})`} />
      <rect x="13" y="776" width="94" height="5" fill="#000" opacity="0.08" />
      <rect x="6" y="781" width="108" height="14" rx="7" fill={`url(#stone-${id})`} />
      <rect x="0" y="795" width="120" height="25" rx="1" fill={`url(#stone-${id})`} />
      <rect x="0" y="795" width="120" height="3" fill="#000" opacity="0.08" />
    </svg>
  );
}
