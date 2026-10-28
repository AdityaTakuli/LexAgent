import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { useRef, type PointerEvent } from "react";

gsap.registerPlugin(useGSAP);

export const prefersReducedMotion = (): boolean =>
  typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Easing curves and durations: quick, settled, no overshoot.
export const M3_EASE = {
  emphasized: [0.2, 0, 0, 1] as [number, number, number, number],
  emphasizedDecelerate: [0.05, 0.7, 0.1, 1] as [number, number, number, number],
  emphasizedAccelerate: [0.3, 0, 0.8, 0.15] as [number, number, number, number],
  standard: [0.2, 0, 0, 1] as [number, number, number, number],
};
export const M3_DURATION = { short: 0.2, medium: 0.35, long: 0.5 };

/** A number that counts up to its value when it appears (GSAP), and snaps straight to it under reduced motion. */
export function CountUp({ value, format, className }: { value: number; format: (n: number) => string; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  useGSAP(
    () => {
      const node = ref.current;
      if (!node) return;
      if (prefersReducedMotion()) {
        node.textContent = format(value);
        return;
      }
      const counter = { v: 0 };
      gsap.to(counter, {
        v: value,
        duration: 1.1,
        ease: "power3.out",
        onUpdate: () => {
          node.textContent = format(counter.v);
        },
        onComplete: () => {
          node.textContent = format(value);
        },
      });
    },
    { dependencies: [value] },
  );
  return (
    <span ref={ref} className={className}>
      {format(value)}
    </span>
  );
}

/** Shared Framer Motion presets. */
export const spring = { type: "spring" as const, stiffness: 420, damping: 34, mass: 0.8 };
export const riseIn = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: M3_DURATION.medium, ease: M3_EASE.emphasizedDecelerate },
};

/** Press feedback hook kept for its call sites. The design uses a quiet highlight (the
 *  .state-layer wash) rather than a spreading ripple, so this renders nothing. */
export function useRipple() {
  return { onPointerDown: undefined as ((event: PointerEvent<HTMLElement>) => void) | undefined, layer: null };
}
