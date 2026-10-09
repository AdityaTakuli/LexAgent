import { useGSAP } from "@gsap/react";
import { AnimatePresence, motion } from "framer-motion";
import gsap from "gsap";
import { useCallback, useRef, useState, type PointerEvent } from "react";

gsap.registerPlugin(useGSAP);

export const prefersReducedMotion = (): boolean =>
  typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Material 3 motion tokens: easing curves and durations.
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

/** Shared Framer Motion presets on the M3 curves. */
export const spring = { type: "spring" as const, stiffness: 420, damping: 34, mass: 0.8 };
export const riseIn = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: M3_DURATION.medium, ease: M3_EASE.emphasizedDecelerate },
};

/** Material ripple: put it inside a positioned, overflow-hidden control. It spreads from the pointer. */
export function useRipple() {
  const [ripples, setRipples] = useState<{ id: number; x: number; y: number; size: number }[]>([]);
  const nextId = useRef(0);
  const onPointerDown = useCallback((event: PointerEvent<HTMLElement>) => {
    if (prefersReducedMotion()) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height) * 2.2;
    const id = nextId.current++;
    setRipples((current) => [...current, { id, x: event.clientX - rect.left, y: event.clientY - rect.top, size }]);
    window.setTimeout(() => setRipples((current) => current.filter((ripple) => ripple.id !== id)), 650);
  }, []);
  const layer = (
    <span className="pointer-events-none absolute inset-0 overflow-hidden rounded-[inherit]" aria-hidden>
      <AnimatePresence>
        {ripples.map((ripple) => (
          <motion.span
            key={ripple.id}
            className="absolute rounded-full bg-current"
            style={{ left: ripple.x - ripple.size / 2, top: ripple.y - ripple.size / 2, width: ripple.size, height: ripple.size }}
            initial={{ scale: 0.1, opacity: 0.16 }}
            animate={{ scale: 1, opacity: 0.1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.45, ease: M3_EASE.standard }}
          />
        ))}
      </AnimatePresence>
    </span>
  );
  return { onPointerDown, layer };
}
