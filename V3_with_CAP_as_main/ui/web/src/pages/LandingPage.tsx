import { useGSAP } from "@gsap/react";
import { motion } from "framer-motion";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { Icon, TIERS } from "../components/ui";
import type { Overview, Status, Tier } from "../lib/api";
import { M3_EASE, prefersReducedMotion } from "../lib/motion";
import justiceUrl from "../assets/img/justice.webp";
import { Column } from "../components/Column";

gsap.registerPlugin(ScrollTrigger);

// ── Content ───────────────────────────────────────────────────────────────────
// Every citation and verdict below is taken from the recorded debate (ui/sample_transcript.json).

const QUESTION = "May law enforcement search the digital contents of a cell phone seized from an arrestee without a warrant?";

type Side = "prosecutor" | "defense";
const SLIPS: { side: Side; name: string; cite: string; year: number; tier: Tier }[] = [
  { side: "prosecutor", name: "Riley v. California", cite: "573 U.S. 373", year: 2014, tier: "SUPPORTED" },
  { side: "defense", name: "Chambers v. Maroney", cite: "399 U.S. 42", year: 1970, tier: "MISATTRIBUTED" },
  { side: "prosecutor", name: "Chimel v. California", cite: "395 U.S. 752", year: 1969, tier: "SUPPORTED" },
  { side: "defense", name: "Riley v. California", cite: "573 U.S. 373", year: 2014, tier: "UNCERTAIN" },
];
const VERDICT = "Searching the digital contents of an arrestee’s cell phone without a warrant is unconstitutional.";

const STEPS: { title: string; body: string; detail: string; icon: string }[] = [
  { title: "Retrieve", icon: "menu_book", body: "Hybrid search (keywords and meaning) pulls the passages of the U.S. Reports that bear on the question.", detail: "8 passages per round" },
  { title: "Argue", icon: "swords", body: "The prosecutor makes the strongest case the passages allow, citing at most two authorities.", detail: "Cites only what was retrieved" },
  { title: "Challenge", icon: "shield", body: "The defense attacks the weak points and brings counter-authority from the same corpus.", detail: "Measured as challenge strength" },
  { title: "Verify", icon: "fact_check", body: "Every citation is looked up in the case catalog and its holding is tested against the opinion it names.", detail: "Five verdicts, below" },
  { title: "Decide", icon: "alt_route", body: "A controller reads the scores and either ends the debate or sends it back for another round.", detail: "Up to three rounds" },
  { title: "Rule", icon: "gavel", body: "The judge writes the answer from the citations that survived, with a confidence grounded in the checks.", detail: "Rejected cases are listed, not hidden" },
];

const SIDE_META: Record<Side, { label: string; color: string; wash: string }> = {
  prosecutor: { label: "Prosecutor", color: "var(--prosecutor)", wash: "var(--prosecutor-wash)" },
  defense: { label: "Defense", color: "var(--defense)", wash: "var(--defense-wash)" },
};

function tierMeta(tier: Tier) {
  return TIERS.find((item) => item.tier === tier) ?? TIERS[4];
}

// ── Hero animation: the citation check ────────────────────────────────────────
function CitationCheck() {
  const scope = useRef<HTMLDivElement>(null);
  const timeline = useRef<gsap.core.Timeline | null>(null);
  const [paused, setPaused] = useState(false);
  const userPaused = useRef(false);
  const reduced = useRef(prefersReducedMotion());

  useGSAP(
    () => {
      if (reduced.current) return; // the static final frame is the default render
      const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.2, defaults: { ease: "power3.out" } });
      // Start: every citation is on the card, waiting; the judge has not ruled.
      tl.set(".cc-row", { opacity: 0.55 })
        .set(".cc-sweep", { scaleX: 0, autoAlpha: 1 })
        .set(".cc-pending", { autoAlpha: 1 })
        .set(".cc-stamp", { autoAlpha: 0, scale: 0.9 })
        .set(".cc-strike", { scaleX: 0 })
        .set(".cc-name", { opacity: 1 })
        .set(".cc-awaiting", { autoAlpha: 1 })
        .set(".cc-word, .cc-basis", { autoAlpha: 0, y: 4 });

      SLIPS.forEach((slip, index) => {
        const row = `.cc-row-${index}`;
        tl.to(row, { opacity: 1, duration: 0.3 }, index === 0 ? 0.6 : ">0.15")
          .to(`${row} .cc-sweep`, { scaleX: 1, duration: 0.9, ease: "power1.inOut" }, "<")
          .to(`${row} .cc-pending`, { autoAlpha: 0, duration: 0.15 }, ">-0.1")
          .to(`${row} .cc-stamp`, { autoAlpha: 1, scale: 1, duration: 0.3, ease: "back.out(2)" }, "<")
          .to(`${row} .cc-sweep`, { autoAlpha: 0, duration: 0.3 }, "<");
        if (slip.tier === "MISATTRIBUTED") {
          tl.to(`${row} .cc-strike`, { scaleX: 1, duration: 0.35, ease: "power2.inOut" }).to(`${row} .cc-name`, { opacity: 0.5, duration: 0.3 }, "<");
        }
      });

      tl.to(".cc-awaiting", { autoAlpha: 0, duration: 0.25 }, "+=0.35")
        .to(".cc-word", { autoAlpha: 1, y: 0, duration: 0.3, stagger: 0.05 }, ">")
        .to(".cc-basis", { autoAlpha: 1, y: 0, duration: 0.3 }, ">-0.1")
        .to({}, { duration: 3.4 })
        .to(".cc-stamp, .cc-word, .cc-basis, .cc-strike", { autoAlpha: 0, duration: 0.35, ease: "power2.in" })
        .to(".cc-name", { opacity: 1, duration: 0.35 }, "<");
      timeline.current = tl;

      // Pause while the card is off screen or the tab is hidden.
      const node = scope.current;
      const observer = new IntersectionObserver(([entry]) => {
        if (userPaused.current) return;
        if (entry.isIntersecting) tl.play();
        else tl.pause();
      });
      if (node) observer.observe(node);
      const onVisibility = () => {
        if (userPaused.current) return;
        if (document.hidden) tl.pause();
        else tl.play();
      };
      document.addEventListener("visibilitychange", onVisibility);
      return () => {
        observer.disconnect();
        document.removeEventListener("visibilitychange", onVisibility);
      };
    },
    { scope },
  );

  const toggle = useCallback(() => {
    const tl = timeline.current;
    if (!tl) return;
    userPaused.current = !userPaused.current;
    if (userPaused.current) tl.pause();
    else tl.play();
    setPaused(userPaused.current);
  }, []);

  return (
    <div ref={scope} className="relative">
      <figure className="card-outlined relative overflow-hidden !rounded-[22px] p-5 shadow-[var(--shadow-2)] sm:p-6" aria-labelledby="cc-caption">
        <figcaption id="cc-caption" className="flex items-center justify-between gap-3">
          <span className="eyebrow text-ink-2">From a recorded debate</span>
          {!reduced.current && (
            <button type="button" onClick={toggle} className="state-layer label-small inline-flex h-8 items-center gap-1.5 rounded-full border border-line px-3 text-ink-2" aria-pressed={paused}>
              <Icon name={paused ? "play_arrow" : "pause"} size={14} />
              {paused ? "Play" : "Pause"}
            </button>
          )}
        </figcaption>

        <p className="body-medium mt-3 rounded-2xl bg-surface-2 px-4 py-3 text-ink">{QUESTION}</p>

        <ol className="mt-4 flex flex-col gap-2" aria-label="Citations and their verdicts">
          {SLIPS.map((slip, index) => {
            const side = SIDE_META[slip.side];
            const tier = tierMeta(slip.tier);
            return (
              <li key={index} className={`cc-row cc-row-${index} relative overflow-hidden rounded-xl border border-line bg-surface-container px-3.5 py-2.5`}>
                <span className="cc-sweep pointer-events-none absolute bottom-0 left-0 h-[2px] w-full origin-left opacity-0" style={{ background: side.color }} aria-hidden />
                <div className="relative flex items-center gap-3">
                  <span className="w-[3px] shrink-0 self-stretch rounded-full" style={{ background: side.color }} aria-hidden />
                  <div className="min-w-0 flex-1">
                    <div className="label-small flex flex-wrap items-baseline gap-x-2">
                      <span style={{ color: side.color }}>{side.label}</span>
                      <span className="cite text-ink-2">
                        {slip.cite} · {slip.year}
                      </span>
                    </div>
                    <div className="relative mt-0.5 inline-block max-w-full">
                      <span className={`cc-name font-display block text-[17px] leading-6 text-ink ${slip.tier === "MISATTRIBUTED" ? "opacity-50" : ""}`}>{slip.name}</span>
                      {slip.tier === "MISATTRIBUTED" && <span className="cc-strike pointer-events-none absolute left-0 right-0 top-1/2 h-px origin-left bg-ink" aria-hidden />}
                    </div>
                  </div>
                  <span className="grid shrink-0 [&>*]:col-start-1 [&>*]:row-start-1 [&>*]:justify-self-end">
                    <span className="cc-pending label-medium invisible inline-flex h-7 items-center gap-1.5 rounded-lg border border-dashed border-line px-2 text-muted opacity-0" aria-hidden>
                      Pending
                    </span>
                    <span className="cc-stamp label-medium inline-flex h-7 items-center gap-1.5 rounded-lg border border-line bg-surface-container px-2 text-ink">
                      <Icon name={tier.icon} size={15} fill style={{ color: tier.color }} />
                      {tier.label}
                    </span>
                  </span>
                </div>
              </li>
            );
          })}
        </ol>

        <div className="mt-4 rounded-2xl border border-line bg-surface px-4 py-3.5">
          <div className="label-small flex items-center gap-1.5 text-ink-2">
            <Icon name="gavel" size={14} /> Judge
          </div>
          <div className="relative mt-1">
            <p className="cc-awaiting body-medium invisible absolute inset-x-0 top-0.5 text-muted opacity-0" aria-hidden>
              Rules once every citation has been checked.
            </p>
            <p className="font-display text-[19px] leading-7 text-ink">
              {VERDICT.split(" ").map((word, index) => (
                <span key={index} className="cc-word inline-block">
                  {word}&nbsp;
                </span>
              ))}
            </p>
            <p className="cc-basis body-small mt-1.5 text-ink-2">
              Relying on Riley v. California, <span className="cite">573 U.S. 373</span> (2014).
            </p>
          </div>
        </div>
      </figure>
    </div>
  );
}

// ── The docket: landmark opinions in the corpus, scrolling past ───────────────
const DOCKET: { name: string; cite: string; year: number }[] = [
  { name: "Brown v. Board of Education", cite: "347 U.S. 483", year: 1954 },
  { name: "Mapp v. Ohio", cite: "367 U.S. 643", year: 1961 },
  { name: "Gideon v. Wainwright", cite: "372 U.S. 335", year: 1963 },
  { name: "Brady v. Maryland", cite: "373 U.S. 83", year: 1963 },
  { name: "New York Times Co. v. Sullivan", cite: "376 U.S. 254", year: 1964 },
  { name: "Griswold v. Connecticut", cite: "381 U.S. 479", year: 1965 },
  { name: "Miranda v. Arizona", cite: "384 U.S. 436", year: 1966 },
  { name: "Katz v. United States", cite: "389 U.S. 347", year: 1967 },
  { name: "Terry v. Ohio", cite: "392 U.S. 1", year: 1968 },
  { name: "Miller v. California", cite: "413 U.S. 15", year: 1973 },
  { name: "Strickland v. Washington", cite: "466 U.S. 668", year: 1984 },
  { name: "Chevron U.S.A. v. NRDC", cite: "467 U.S. 837", year: 1984 },
  { name: "Batson v. Kentucky", cite: "476 U.S. 79", year: 1986 },
  { name: "Daubert v. Merrell Dow", cite: "509 U.S. 579", year: 1993 },
  { name: "United States v. Lopez", cite: "514 U.S. 549", year: 1995 },
  { name: "Kyllo v. United States", cite: "533 U.S. 27", year: 2001 },
  { name: "Roper v. Simmons", cite: "543 U.S. 551", year: 2005 },
  { name: "District of Columbia v. Heller", cite: "554 U.S. 570", year: 2008 },
  { name: "Citizens United v. FEC", cite: "558 U.S. 310", year: 2010 },
  { name: "Graham v. Florida", cite: "560 U.S. 48", year: 2010 },
  { name: "Miller v. Alabama", cite: "567 U.S. 460", year: 2012 },
  { name: "Riley v. California", cite: "573 U.S. 373", year: 2014 },
  { name: "Obergefell v. Hodges", cite: "576 U.S. 644", year: 2015 },
  { name: "Carpenter v. United States", cite: "585 U.S. 296", year: 2018 },
];

function DocketRow({ items, reverse, duration }: { items: typeof DOCKET; reverse?: boolean; duration: number }) {
  const scope = useRef<HTMLDivElement>(null);
  const tween = useRef<gsap.core.Tween | null>(null);
  useGSAP(
    () => {
      if (prefersReducedMotion()) return;
      const track = scope.current?.querySelector("ul");
      if (!track) return;
      tween.current = gsap.fromTo(track, { xPercent: reverse ? -50 : 0 }, { xPercent: reverse ? 0 : -50, duration, ease: "none", repeat: -1 });
    },
    { scope },
  );
  // Ease to a stop on hover so a name can be read, then pick the pace back up.
  const slow = (to: number) => tween.current && gsap.to(tween.current, { timeScale: to, duration: 0.6, ease: "power2.out" });
  const loop = [...items, ...items];
  return (
    <div ref={scope} className="overflow-hidden [mask-image:linear-gradient(90deg,transparent,#000_8%,#000_92%,transparent)]" onMouseEnter={() => slow(0)} onMouseLeave={() => slow(1)}>
      <ul className="flex w-max gap-3 pr-3">
        {loop.map((item, index) => (
          <li key={index} aria-hidden={index >= items.length} className="flex shrink-0 items-baseline gap-2.5 rounded-full border border-line bg-surface-container px-4 py-2">
            <span className="font-display text-[16px] text-ink">{item.name}</span>
            <span className="cite text-ink-2">
              {item.cite} ({item.year})
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Docket() {
  const half = Math.ceil(DOCKET.length / 2);
  return (
    <section aria-labelledby="docket-title" className="border-t border-line py-10">
      <div className="mx-auto mb-6 flex max-w-6xl items-baseline justify-between gap-4 px-5">
        <h2 id="docket-title" className="eyebrow text-ink-2">
          In the corpus
        </h2>
        <p className="body-small text-ink-2">Landmark opinions LexAgent reads and checks against, among 9,333 reports.</p>
      </div>
      <div className="flex flex-col gap-3">
        <DocketRow items={DOCKET.slice(0, half)} duration={70} />
        <DocketRow items={DOCKET.slice(half)} duration={80} reverse />
      </div>
    </section>
  );
}

// ── Sections ──────────────────────────────────────────────────────────────────
function Reveal({ children, className = "", delay = 0 }: { children: ReactNode; className?: string; delay?: number }) {
  return (
    <motion.div initial={{ opacity: 0, y: 18 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-80px" }} transition={{ duration: 0.6, delay, ease: M3_EASE.emphasizedDecelerate }} className={className}>
      {children}
    </motion.div>
  );
}

function SectionHeading({ eyebrow, title, lead }: { eyebrow: string; title: ReactNode; lead?: ReactNode }) {
  return (
    <Reveal className="max-w-2xl">
      <p className="eyebrow text-ink-2">{eyebrow}</p>
      <h2 className="display-medium mt-3 text-ink">{title}</h2>
      {lead && <p className="body-large mt-4 text-ink-2">{lead}</p>}
    </Reveal>
  );
}

function Steps() {
  const scope = useRef<HTMLOListElement>(null);
  useGSAP(
    () => {
      if (prefersReducedMotion()) return;
      gsap.fromTo(".steps-progress", { scaleY: 0 }, { scaleY: 1, ease: "none", scrollTrigger: { trigger: scope.current, start: "top 70%", end: "bottom 60%", scrub: 0.4 } });
      gsap.utils.toArray<HTMLElement>(".step-dot").forEach((dot) => {
        ScrollTrigger.create({ trigger: dot, start: "top 62%", toggleClass: { targets: dot, className: "is-reached" } });
      });
    },
    { scope },
  );
  return (
    <ol ref={scope} className="relative mt-12 grid gap-10 pl-10 sm:pl-14">
      <span className="absolute bottom-2 left-[11px] top-2 w-px bg-line sm:left-[15px]" aria-hidden />
      <span className="steps-progress absolute bottom-2 left-[11px] top-2 w-px origin-top bg-ink sm:left-[15px]" aria-hidden />
      {STEPS.map((step, index) => (
        <li key={step.title} className="relative">
          <span className="step-dot absolute -left-10 top-1 h-[23px] w-[23px] rounded-full border border-ink bg-[var(--raised)] transition-colors duration-300 sm:-left-14 sm:h-[31px] sm:w-[31px] [&.is-reached]:bg-ink" aria-hidden />
          <Reveal className="grid gap-2 md:grid-cols-[220px_1fr_auto] md:items-baseline md:gap-8">
            <h3 className="flex items-baseline gap-3">
              <span className="cite text-ink-2">{String(index + 1).padStart(2, "0")}</span>
              <span className="font-display text-[26px] leading-8 text-ink">{step.title}</span>
            </h3>
            <p className="body-large max-w-xl text-ink-2">{step.body}</p>
            <p className="label-medium flex items-center gap-1.5 text-ink-2">
              <Icon name={step.icon} size={16} />
              {step.detail}
            </p>
          </Reveal>
        </li>
      ))}
    </ol>
  );
}

function Verdicts() {
  return (
    <div className="mt-12 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
      {TIERS.map((tier, index) => (
        <Reveal key={tier.tier} delay={index * 0.05} className="card-outlined flex flex-col p-5">
          <Icon name={tier.icon} size={22} fill style={{ color: tier.color }} />
          <h3 className="title-medium mt-4 text-ink">{tier.label}</h3>
          <p className="body-small mt-1.5 text-ink-2">{tier.description}</p>
        </Reveal>
      ))}
    </div>
  );
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Reveal className="border-t border-line pt-5">
      <dt className="eyebrow text-ink-2">{label}</dt>
      <dd className="body-large mt-2 text-ink">{children}</dd>
    </Reveal>
  );
}

// ── Hero: Lady Justice on the law books, between two columns, in front of the name ──
const HERO_WORD = "LEXAGENT";

function Hero({ live, chunks, volumes }: { live: boolean; chunks: number; volumes: number }) {
  const scope = useRef<HTMLElement>(null);
  useGSAP(
    () => {
      if (prefersReducedMotion()) return;
      const tl = gsap.timeline({ defaults: { ease: "expo.out" } });
      tl.from(".hero-col", { yPercent: 100, duration: 1.5, stagger: 0.1 }, 0)
        .from(".hero-letter", { yPercent: 105, duration: 1.1, stagger: 0.045 }, 0.25)
        .from(".hero-statue", { yPercent: 12, autoAlpha: 0, duration: 1.6 }, 0.5)
        .from(".hero-copy", { y: 14, autoAlpha: 0, duration: 0.8, stagger: 0.08, ease: "power3.out" }, 0.4);
      // On scroll the figure lifts a little and the name sinks: a hint of depth.
      const st = { trigger: scope.current, start: "top top", end: "bottom top", scrub: true };
      gsap.to(".hero-figure-inner", { yPercent: -4, ease: "none", scrollTrigger: st });
      gsap.to(".hero-word", { yPercent: 22, ease: "none", scrollTrigger: st });
    },
    { scope },
  );

  const button = "label-large inline-flex h-12 items-center gap-2 rounded-full px-6";
  return (
    <section ref={scope} className="hero-sage relative isolate -mt-16 flex flex-col overflow-hidden pt-16 xl:block xl:h-[max(880px,min(100svh,1000px))]">
      <div className="hero-grain pointer-events-none absolute inset-0" aria-hidden />
      <div className="hero-gridlines pointer-events-none absolute inset-0" aria-hidden />

      {/* Columns, framing the scene like a courthouse portico */}
      <div className="pointer-events-none absolute bottom-0 left-[1.5vw] hidden h-[86%] items-end xl:flex" aria-hidden>
        <Column className="hero-col h-full w-auto drop-shadow-[8px_18px_24px_rgba(29,26,21,0.28)]" />
      </div>
      <div className="pointer-events-none absolute bottom-0 right-[1.5vw] hidden h-[86%] -scale-x-100 items-end xl:flex" aria-hidden>
        <Column className="hero-col h-full w-auto drop-shadow-[8px_18px_24px_rgba(29,26,21,0.28)]" />
      </div>

      {/* Copy: headline (top left on wide screens) */}
      <div className="hero-copy relative z-30 order-1 px-5 pt-8 sm:px-8 xl:absolute xl:bottom-[18%] xl:left-[max(170px,12vw)] xl:w-[min(400px,25vw)] xl:p-0">
        <p className="eyebrow text-ink">Multi-agent legal research · U.S. Supreme Court</p>
        <h1 className="mt-3 font-display text-[40px] leading-[1.05] tracking-[-0.015em] text-ink xl:text-[48px]">
          Legal answers that <em className="italic">show their work.</em>
        </h1>
        <p className="body-medium mt-4 text-ink">
          Two AI advocates argue your question from Supreme Court opinions. A <mark className="hero-mark">prosecutor</mark> makes the case, a{" "}
          <mark className="hero-mark">defense</mark> attacks it, every case they cite is checked against the opinion it names, and a judge rules only on what survives.
        </p>
          <div className="mt-6 flex flex-wrap items-center gap-3 xl:w-max xl:flex-nowrap">
            <a href="#/chat" className={`${button} bg-accent text-accent-ink shadow-[var(--shadow-2)] transition-transform hover:-translate-y-px`}>
              Ask a question <Icon name="arrow_forward" size={17} />
            </a>
            <a href="#/chat?replay=1" className={`${button} state-layer border border-ink/25 bg-[#f3ecdf]/70 text-ink`}>
              <Icon name="play_arrow" size={16} /> Watch a debate
            </a>
          </div>
          <p className="body-small mt-3 text-ink" role="status">
            {live ? "Live now. A debate takes about a minute." : "Live questions are paused right now; recorded debates still play."}
          </p>
      </div>

      {/* The scene: the name behind, the statue standing on the books in front */}
      <div className="relative order-3 mt-6 h-[min(620px,150vw)] xl:absolute xl:inset-x-0 xl:bottom-0 xl:mt-0 xl:h-[90%]" aria-hidden>
        <div className="hero-word pointer-events-none absolute inset-x-0 top-[1%] flex select-none justify-center">
          <span className="font-display text-[14.5vw] uppercase leading-[0.78] tracking-[-0.01em] text-ink xl:text-[min(13vw,250px)]">
            {HERO_WORD.split("").map((letter, index) => (
              <span key={index} className="inline-block overflow-hidden align-bottom">
                <span className="hero-letter inline-block">{letter}</span>
              </span>
            ))}
          </span>
        </div>
        <div className="hero-figure-inner absolute inset-0">
          <div className="absolute bottom-0 left-1/2 h-[88%] -translate-x-1/2">
            <img
              src={justiceUrl}
              alt=""
              width={726}
              height={1483}
              decoding="async"
              className="hero-statue h-full w-auto max-w-none drop-shadow-[0_18px_22px_rgba(29,26,21,0.35)]"
            />
          </div>
        </div>
      </div>

      {/* Quote, facts and note (lower right) */}
      <div className="hero-copy relative z-30 order-4 px-5 pb-10 pt-10 sm:px-8 xl:absolute xl:bottom-[6%] xl:right-[max(170px,12vw)] xl:w-[min(340px,22vw)] xl:p-0 xl:text-right">
        <figure>
          <blockquote className="font-display text-[20px] italic leading-[29px] text-ink">
            “Cross-examination is beyond any doubt the greatest legal engine ever invented for the discovery of truth.”
          </blockquote>
          <figcaption className="body-small mt-3 text-ink">
            John Henry Wigmore, quoted by the Supreme Court in <span className="italic">California v. Green</span>, <span className="cite">399 U.S. 149</span> (1970)
          </figcaption>
        </figure>
        <p className="body-small mt-6 border-t border-ink/25 pt-4 text-ink">
          {chunks.toLocaleString("en-US")} passages from {volumes} volumes of the U.S. Reports, every citation checked against the opinion it names. A research prototype; answers are AI-written and are not legal advice.
        </p>
      </div>
    </section>
  );
}

function LiveCheck() {
  return (
    <section className="border-t border-line">
      <div className="mx-auto grid max-w-6xl items-center gap-12 px-5 py-20 lg:grid-cols-[0.9fr_1.1fr] lg:gap-16 lg:py-28">
        <SectionHeading
          eyebrow="The check, as it happens"
          title={
            <>
              Every citation is <em className="italic">checked</em> before the judge rules.
            </>
          }
          lead={
            <>
              In this recorded debate on cell-phone searches, the prosecutor’s authorities held up. The defense cited <span className="italic">Chambers v. Maroney</span> for a holding its opinion does not contain, and the check set it aside.
            </>
          }
        />
        <Reveal className="min-w-0">
          <CitationCheck />
        </Reveal>
      </div>
    </section>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export function LandingPage({ status, overview }: { status: Status | null; overview: Overview | null }) {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  useEffect(() => {
    document.documentElement.style.overflowY = "auto";
    document.getElementById("root")?.style.setProperty("height", "auto");
    return () => {
      document.documentElement.style.overflowY = "";
      document.getElementById("root")?.style.removeProperty("height");
    };
  }, []);

  const scrollTo = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth" });
  const live = status !== null;
  const chunks = status?.corpus.chunks || overview?.corpus.chunks || 26724;
  const volumes = status?.corpus.volumes_indexed || overview?.corpus.volumes_indexed || 27;
  const generator = overview?.corpus.models.generator ?? "gpt-oss-120b";

  return (
    <div className="relative min-h-full bg-bg text-ink">
      {/* Navigation bar */}
      <header className={`sticky top-0 z-30 transition-[background,border-color,backdrop-filter] ${scrolled ? "glass border-x-0 border-t-0" : "border-b border-transparent"}`}>
        <nav className="flex h-16 w-full items-center justify-between px-5 sm:px-8 lg:px-12" aria-label="Main">
          <a href="#/" className="flex items-center gap-2 text-ink" aria-label="LexAgent home">
            <Icon name="balance" size={19} />
            <span className="font-display text-[21px] leading-none tracking-[-0.01em]">LexAgent</span>
          </a>
          <div className="flex items-center gap-1 sm:gap-2">
            <button type="button" onClick={() => scrollTo("how")} className="label-medium hidden h-9 rounded-full px-3 text-ink hover:underline sm:inline-flex sm:items-center">
              How it works
            </button>
            <button type="button" onClick={() => scrollTo("verdicts")} className="label-medium hidden h-9 rounded-full px-3 text-ink hover:underline sm:inline-flex sm:items-center">
              Citation check
            </button>
            <a href="#/chat" className="label-medium inline-flex h-9 items-center gap-1.5 rounded-full bg-accent px-4 text-accent-ink">
              Open LexAgent
            </a>
          </div>
        </nav>
      </header>

      <main className="relative">
        <Hero live={live} chunks={chunks} volumes={volumes} />

        <Docket />

        <LiveCheck />

        {/* How it works */}
        <section id="how" className="scroll-mt-16 border-t border-line">
          <div className="mx-auto max-w-6xl px-5 py-20 lg:py-28">
            <SectionHeading
              eyebrow="How a question is decided"
              title={
                <>
                  An argument, a rebuttal, <em className="italic">then the evidence.</em>
                </>
              }
              lead="One model answering alone will cite cases that do not exist. LexAgent makes the answer survive an adversary and a citation check first."
            />
            <Steps />
          </div>
        </section>

        {/* Verdicts */}
        <section id="verdicts" className="scroll-mt-16 border-t border-line">
          <div className="mx-auto max-w-6xl px-5 py-20 lg:py-28">
            <SectionHeading
              eyebrow="The citation check"
              title="Five verdicts for every case cited."
              lead={
                <>
                  Each citation is looked up in the case catalog, then its holding is tested against the opinion it names. In the recorded debate above, the defense cited <span className="italic">Chambers v. Maroney</span> for a holding its opinion does not contain, and the check marked it Misattributed.
                </>
              }
            />
            <Verdicts />
          </div>
        </section>

        {/* What's inside */}
        <section className="border-t border-line">
          <div className="mx-auto max-w-6xl px-5 py-20 lg:py-28">
            <SectionHeading eyebrow="What it reads" title="The U.S. Reports, from Brown to the present." />
            <dl className="mt-12 grid gap-x-10 gap-y-8 md:grid-cols-3">
              <Fact label="Corpus">
                {chunks.toLocaleString("en-US")} passages from {volumes} volumes of the U.S. Reports, 1954 to 2025: Harvard’s Caselaw Access Project, with the newest volumes from the Court’s official bound volumes.
              </Fact>
              <Fact label="Models">
                Arguments and rulings by {generator}. Citations are checked by Legal-BERT and an entailment model running on LexAgent’s own server.
              </Fact>
              <Fact label="Debate">Up to three rounds. The controller ends early when the citations hold and sends the debate back when they do not.</Fact>
            </dl>
          </div>
        </section>

        {/* Limits */}
        <section className="border-t border-line">
          <div className="mx-auto grid max-w-6xl gap-10 px-5 py-20 md:grid-cols-[1fr_1.2fr] lg:py-24">
            <SectionHeading eyebrow="Before you rely on it" title="What LexAgent is not." />
            <Reveal className="body-large flex flex-col gap-5 text-ink-2">
              <p>
                <span className="font-medium text-ink">Not legal advice.</span> LexAgent is a research prototype. Answers are written by an AI model and can be wrong, even after the citation check. Read the opinions it cites.
              </p>
              <p>
                <span className="font-medium text-ink">Not private.</span> Questions are sent to third-party model providers to be answered. Leave out names, case details and anything confidential.
              </p>
              <p>
                <span className="font-medium text-ink">Not every court.</span> It reads U.S. Supreme Court opinions in its corpus and nothing else, so questions of state law or recent lower-court rulings are out of scope.
              </p>
            </Reveal>
          </div>
        </section>

        {/* Closing call to action */}
        <section className="border-t border-line bg-inverse-surface text-inverse-on-surface">
          <div className="mx-auto flex max-w-6xl flex-col items-start gap-8 px-5 py-20 md:flex-row md:items-center md:justify-between lg:py-24">
            <Reveal>
              <h2 className="display-medium max-w-xl">Put a question to the Court’s own words.</h2>
            </Reveal>
            <Reveal delay={0.1} className="flex flex-wrap gap-3">
              <a href="#/chat" className="label-large inline-flex h-12 items-center gap-2 rounded-full bg-inverse-on-surface px-6 text-inverse-surface">
                Ask a question <Icon name="arrow_forward" size={17} />
              </a>
              <a href="#/chat?replay=1" className="label-large inline-flex h-12 items-center gap-2 rounded-full border border-current/25 px-6">
                <Icon name="play_arrow" size={16} /> Watch a debate
              </a>
            </Reveal>
          </div>
        </section>
      </main>

      <footer className="border-t border-line">
        <div className="body-small mx-auto flex max-w-6xl flex-col gap-2 px-5 py-8 text-ink-2 sm:flex-row sm:justify-between">
          <span>© {new Date().getFullYear()} Velocity Technologies. LexAgent is a research prototype.</span>
          <span>Opinions: Caselaw Access Project (Harvard Law School Library) and the Supreme Court of the United States.</span>
        </div>
      </footer>
    </div>
  );
}
