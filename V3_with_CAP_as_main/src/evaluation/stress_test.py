# LexAgent v3.0 | stress_test.py
"""
Citation Stress-Test Protocol (Novelty 5).
A controlled, zero-LLM diagnostic suite that feeds 6 distinct classes of labelled
citations directly into the Citation Confidence Engine (CCE) verifier:
  1. Correct: Target case + true holding (Expected: SUPPORTED)
  2. Misattributed: Target case + holding from different case (Expected: MISATTRIBUTED)
  3. CaseHOLD distractor: Target case + plausible hard negative holding (Expected: MISATTRIBUTED)
  4. Fabricated with cite: Invented case name with cite in covered volume (Expected: FABRICATED)
  5. Fabricated without cite: Invented case name with no cite (Expected: UNVERIFIABLE)
  6. Real off-corpus: Real landmark case outside covered volumes (Expected: UNVERIFIABLE)
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional

from src.data.corpus_schema import CitationRecord
from src.novel.cce import CitationConfidenceEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Real landmark cases that are NOT in the 27 Landmark CAP Volumes (Volumes 347, 410, 585, etc.)
OFF_CORPUS_LANDMARK_CASES = [
    {
        "case_name": "Marbury v. Madison",
        "reporter_cite": "5 U.S. 137",
        "year": 1803,
        "holding": "It is emphatically the province and duty of the judicial department to say what the law is, establishing judicial review.",
    },
    {
        "case_name": "McCulloch v. Maryland",
        "reporter_cite": "17 U.S. 316",
        "year": 1819,
        "holding": "Congress has implied powers under the Necessary and Proper Clause to create the Second Bank of the United States.",
    },
    {
        "case_name": "Gibbons v. Ogden",
        "reporter_cite": "22 U.S. 1",
        "year": 1824,
        "holding": "The Commerce Clause of the Constitution grants the federal government the power to regulate interstate commerce.",
    },
    {
        "case_name": "Barron v. Baltimore",
        "reporter_cite": "32 U.S. 243",
        "year": 1833,
        "holding": "The Bill of Rights restricts only the actions of the federal government, not state governments.",
    },
    {
        "case_name": "Dred Scott v. Sandford",
        "reporter_cite": "60 U.S. 393",
        "year": 1857,
        "holding": "Enslaved African Americans were not citizens under the Constitution and therefore had no standing to sue in federal court.",
    },
    {
        "case_name": "Ex parte Milligan",
        "reporter_cite": "71 U.S. 2",
        "year": 1866,
        "holding": "Military tribunals cannot try citizens in states where civil courts are open and functioning.",
    },
    {
        "case_name": "Civil Rights Cases",
        "reporter_cite": "109 U.S. 3",
        "year": 1883,
        "holding": "The Fourteenth Amendment does not permit federal legislation prohibiting racial discrimination by private individuals.",
    },
    {
        "case_name": "Plessy v. Ferguson",
        "reporter_cite": "163 U.S. 537",
        "year": 1896,
        "holding": "Racial segregation laws for public facilities do not violate the Fourteenth Amendment so long as the segregated facilities are equal.",
    },
    {
        "case_name": "Lochner v. New York",
        "reporter_cite": "198 U.S. 45",
        "year": 1905,
        "holding": "Limits on working hours for bakers violate the freedom of contract protected by the Fourteenth Amendment's Due Process Clause.",
    },
    {
        "case_name": "Schenck v. United States",
        "reporter_cite": "249 U.S. 47",
        "year": 1919,
        "holding": "Speech creating a clear and present danger of substantive evils that Congress may prevent is not protected by the First Amendment.",
    },
    {
        "case_name": "Near v. Minnesota",
        "reporter_cite": "283 U.S. 697",
        "year": 1931,
        "holding": "Prior restraints on publication by the state violate the First and Fourteenth Amendments.",
    },
    {
        "case_name": "Wickard v. Filburn",
        "reporter_cite": "317 U.S. 111",
        "year": 1942,
        "holding": "Congress can regulate purely intrastate, non-commercial production of wheat if the cumulative aggregate affects interstate commerce.",
    },
    {
        "case_name": "Korematsu v. United States",
        "reporter_cite": "323 U.S. 214",
        "year": 1944,
        "holding": "Compulsory exclusion of citizens of Japanese ancestry into internment areas was justified under wartime military necessity.",
    },
    {
        "case_name": "Loving v. Virginia",
        "reporter_cite": "388 U.S. 1",
        "year": 1967,
        "holding": "State anti-miscegenation statutes prohibiting interracial marriage violate both the Equal Protection and Due Process Clauses.",
    },
    {
        "case_name": "Tinker v. Des Moines Independent Community School District",
        "reporter_cite": "393 U.S. 503",
        "year": 1969,
        "holding": "Students do not shed their constitutional rights to freedom of speech or expression at the schoolhouse gate.",
    },
    {
        "case_name": "Gideon v. Wainwright",
        "reporter_cite": "372 U.S. 335",
        "year": 1963,
        "holding": "The Fourteenth Amendment incorporates the Sixth Amendment right to counsel for indigent defendants in state felony trials.",
    },
    {
        "case_name": "New York Times Co. v. Sullivan",
        "reporter_cite": "376 U.S. 254",
        "year": 1964,
        "holding": "Public officials cannot recover damages for defamatory falsehoods absent proof of actual malice with knowledge or reckless disregard.",
    },
    {
        "case_name": "Terry v. Ohio",
        "reporter_cite": "392 U.S. 1",
        "year": 1968,
        "holding": "Police may perform a protective stop and frisk based on reasonable articulable suspicion that criminal activity is afoot.",
    },
    {
        "case_name": "United States v. Nixon",
        "reporter_cite": "418 U.S. 683",
        "year": 1974,
        "holding": "Presidential executive privilege is not absolute and cannot withstand a subpoena duces tecum in a criminal prosecution.",
    },
    {
        "case_name": "Chevron U.S.A. Inc. v. Natural Resources Defense Council",
        "reporter_cite": "467 U.S. 837",
        "year": 1984,
        "holding": "Courts must defer to administrative agencies' reasonable interpretations of ambiguous statutory provisions.",
    },
    {
        "case_name": "Texas v. Johnson",
        "reporter_cite": "491 U.S. 397",
        "year": 1989,
        "holding": "Desecration of the American flag constitutes protected symbolic speech under the First Amendment.",
    },
    {
        "case_name": "Planned Parenthood of Southeastern Pennsylvania v. Casey",
        "reporter_cite": "505 U.S. 833",
        "year": 1992,
        "holding": "State regulations on pre-viability abortion are invalid if they place an undue burden on a woman's reproductive choices.",
    },
    {
        "case_name": "Lawrence v. Texas",
        "reporter_cite": "539 U.S. 558",
        "year": 2003,
        "holding": "State criminalization of consensual adult intimate sodomy violates the Due Process Clause of the Fourteenth Amendment.",
    },
    {
        "case_name": "District of Columbia v. Heller",
        "reporter_cite": "554 U.S. 570",
        "year": 2008,
        "holding": "The Second Amendment guarantees an individual right to possess a firearm unconnected with service in a militia for self-defense.",
    },
    {
        "case_name": "Citizens United v. Federal Election Commission",
        "reporter_cite": "558 U.S. 310",
        "year": 2010,
        "holding": "Political spending is a protected form of speech under the First Amendment that cannot be restricted for corporations or unions.",
    },
]

# Hard negative holdings (distractors) across diverse legal domains
HARD_NEGATIVE_DISTRACTORS = [
    "Police officers may conduct a full warrantless search of any arrestee's home without probable cause or exigent circumstances.",
    "The Eighth Amendment permits mandatory death sentences for all non-violent property crimes committed by minors under 16.",
    "State courts are not required to provide legal counsel to indigent criminal defendants in capital or felony trials.",
    "Prosecutors have no constitutional duty to disclose exculpatory evidence requested by defense counsel under the Due Process Clause.",
    "Private citizens are strictly subject to Fourth Amendment warrant requirements when investigating neighbors or reporting crimes.",
    "The First Amendment permits municipal governments to ban political demonstrations criticizing sitting elected officials.",
    "The Fifth Amendment double jeopardy protection does not apply to state court prosecutions following federal acquittals.",
    "Schools possess unfettered authority to censor student political speech off-campus without showing substantial disruption.",
    "Commercial corporations possess absolute sovereign immunity from federal antitrust enforcement under the Sherman Act.",
    "Federal regulatory agencies may alter statutory text without congressional approval or notice-and-comment rulemaking.",
    "Criminal defendants may be tried in absentia without notice or an opportunity to confront adverse witnesses.",
    "The Dormant Commerce Clause allows states to impose discriminatory protective tariffs on goods imported from sister states.",
    "Presidential executive orders supersede ratified treaties and constitutional statutory entitlements without judicial review.",
    "Public universities may condition academic degrees on compulsory religious affiliation and mandatory theological oaths.",
    "A municipality may seize private real property for the sole purpose of gifting it to a private corporate executive without compensation.",
    "Pre-trial detainees can be detained indefinitely without a bail hearing or indictment under standard federal criminal procedure.",
    "Law enforcement officers may use deadly force to prevent any non-violent misdemeanant from walking away from an investigatory stop.",
    "Sentencing judges have mandatory discretion to disregard statutory minimum sentences established by Congress in narcotics statutes.",
    "Contractual arbitration clauses are per se unconstitutional under the Seventh Amendment right to a civil jury trial.",
    "Copyright protections expire immediately upon electronic dissemination over public internet networks.",
]

# Fabricated case names with citations matching federal volume formats
INVENTED_PARTY_NAMES = [
    ("Consolidated Steel Logistics v. Apex Transport", "410 U.S. 995"),
    ("United Maritime Carriers v. Federal Commerce Commission", "347 U.S. 981"),
    ("Quantum Biometrics Corp v. State Department of Revenue", "585 U.S. 991"),
    ("Meridian National Bank v. Global Mining Enterprises", "372 U.S. 988"),
    ("Vanguard Telecommunications v. Pacific Rail Authorities", "384 U.S. 977"),
    ("Aegis Pharmaceutical Solutions v. United Food & Drug Board", "410 U.S. 997"),
    ("Borealis Energy Infrastructure v. Environmental Safety Commission", "585 U.S. 994"),
    ("Cascade Timberland Holdings v. National Forestry Service", "347 U.S. 986"),
    ("Delta Aerospace Systems v. Department of Defense Procurement", "372 U.S. 992"),
    ("Echelon Financial Derivatives v. Securities Exchange Authority", "384 U.S. 982"),
    ("Frontier Satellite Networks v. Federal Communications Board", "410 U.S. 992"),
    ("Genesis Biotechnology Laboratories v. Department of Health", "585 U.S. 998"),
    ("Horizon Container Freightways v. Interstate Transit Council", "347 U.S. 993"),
    ("Ironclad Security Protocols v. Cyber Operations Directorate", "372 U.S. 995"),
    ("Jupiter Mining & Refining v. Department of the Interior", "384 U.S. 989"),
    ("Kinetix Logistics Group v. National Highway Safety Administration", "410 U.S. 989"),
    ("Liberty Maritime Dredging v. Army Corps of Engineers", "585 U.S. 992"),
    ("Monarch Chemical Synthetics v. Environmental Oversight Agency", "347 U.S. 989"),
    ("Northstar Nuclear Power v. Atomic Energy Regulatory Commission", "372 U.S. 999"),
    ("Olympus Software Architecture v. Patent Examination Office", "384 U.S. 993"),
]

INVENTED_NO_CITE_NAMES = [
    "CyberCorp Dynamics v. Hyperion Aerospace",
    "OmniGlobal Media Group v. Vertex Holdings",
    "Atlas Artificial Intelligence Inc v. State of Columbia",
    "Pinnacle Pharmaceutical Consortium v. United Health Alliance",
    "Nexus Robotics v. Pacific Coast Harbor Commission",
    "AeroTech Propulsion v. National Aerospace Directorate",
    "BioGenesis Synthetics v. Federal Bioethics Council",
    "Centurion Capital Management v. Global Markets Tribunal",
    "Dynamix Energy Systems v. Department of Energy Oversight",
    "Equinox Semiconductor Corp v. Bureau of Industry and Trade",
    "Fortress Cloud Infrastructure v. Cybersecurity Enforcement Board",
    "Global Trade Syndicate v. International Commerce Commission",
    "Halcyon Medical Devices v. Healthcare Regulatory Agency",
    "Infinitum Quantum Computing v. Department of Science and Tech",
    "JurisTech Legal Automation v. Federal Bar Accreditation Committee",
    "Kronos Logistics Worldwide v. Surface Transportation Directorate",
    "Luminary Optics Technologies v. Defense Advanced Projects Board",
    "Matrix Distributed Ledger Systems v. Commodity Trading Council",
    "Nova Renewable Resources v. Clean Power Authority",
    "Orbital Telecommunications v. Space Commerce Commission",
]


def build_stress_test_dataset(benchmark_cases: list) -> List[Dict[str, Any]]:
    """Builds synthetic labelled test dataset with 6 controlled citation types."""
    dataset = []

    # 1. Correct Citations (Target case + true holding) - all benchmark cases, no limit
    for case in benchmark_cases:
        target = case.get("target_case", "")
        holding = case.get("ground_truth_holding", "")
        cite = case.get("reporter_citation", "")
        if target and holding:
            dataset.append({
                "type": "correct",
                "label": "SUPPORTED",
                "case_name": target,
                "reporter_cite": cite,
                "holding": holding,
                "description": f"True holding for {target}",
            })

    # 2. Misattributed Citations (Target case + holding from different benchmark case) - all benchmark cases
    n_cases = len(benchmark_cases)
    shift = max(1, n_cases // 2) if n_cases > 0 else 1
    for i in range(n_cases):
        target = benchmark_cases[i].get("target_case", "")
        cite = benchmark_cases[i].get("reporter_citation", "")
        swapped_case = benchmark_cases[(i + shift) % n_cases]
        swapped_holding = swapped_case.get("ground_truth_holding", "")
        if target and swapped_holding:
            dataset.append({
                "type": "misattributed",
                "label": "REJECT",
                "expected_verdicts": ["MISATTRIBUTED", "UNCERTAIN"],
                "case_name": target,
                "reporter_cite": cite,
                "holding": swapped_holding,
                "description": f"{target} paired with holding of {swapped_case.get('target_case', '')}",
            })

    # 3. CaseHOLD Distractor Citations (Target case + hard negative distractor) - all distractors
    for i, distractor in enumerate(HARD_NEGATIVE_DISTRACTORS):
        if benchmark_cases:
            target_case = benchmark_cases[i % len(benchmark_cases)]
            target = target_case.get("target_case", "")
            cite = target_case.get("reporter_citation", "")
            dataset.append({
                "type": "casehold_distractor",
                "label": "REJECT",
                "expected_verdicts": ["MISATTRIBUTED", "UNCERTAIN"],
                "case_name": target,
                "reporter_cite": cite,
                "holding": distractor,
                "description": f"{target} paired with negative distractor #{i+1}",
            })

    # 4. Fabricated With Citation (Invented name + reporter cite in covered volume)
    for name, cite in INVENTED_PARTY_NAMES:
        dataset.append({
            "type": "fabricated_with_cite",
            "label": "FABRICATED",
            "expected_verdicts": ["FABRICATED"],
            "case_name": name,
            "reporter_cite": cite,
            "holding": "Invented proposition regarding regulatory compliance.",
            "description": f"Fabricated case in covered volume: {cite}",
        })

    # 5. Fabricated Without Citation (Invented name, no reporter cite)
    for name in INVENTED_NO_CITE_NAMES:
        dataset.append({
            "type": "fabricated_no_cite",
            "label": "UNVERIFIABLE",
            "expected_verdicts": ["UNVERIFIABLE"],
            "case_name": name,
            "reporter_cite": "",
            "holding": "Unsubstantiated statutory interpretation.",
            "description": f"Fabricated case with no reporter volume: {name}",
        })

    # 6. Real Off-Corpus Landmark Cases (Real cases outside covered volumes)
    for off in OFF_CORPUS_LANDMARK_CASES:
        dataset.append({
            "type": "real_off_corpus",
            "label": "UNVERIFIABLE",
            "expected_verdicts": ["UNVERIFIABLE"],
            "case_name": off["case_name"],
            "reporter_cite": off["reporter_cite"],
            "holding": off["holding"],
            "description": f"Real landmark case outside covered corpus: {off['case_name']}",
        })

    return dataset


def run_citation_stress_test(cce: CitationConfidenceEngine, benchmark_cases: list) -> dict:
    """
    Executes the Citation Stress-Test Protocol.
    Feeds synthetic labelled citations straight into factorized CCE and reports
    per-type catch rates, false acceptance rate, and power.
    """
    logger.info("Executing Citation Stress-Test Protocol across 6 citation classes...")
    start_time = time.time()

    dataset = build_stress_test_dataset(benchmark_cases)
    per_type_counts: Dict[str, Dict[str, int]] = {}
    detailed_results = []

    for item in dataset:
        c_type = item["type"]
        if c_type not in per_type_counts:
            per_type_counts[c_type] = {
                "total": 0, "accepted": 0, "rejected": 0,
                "SUPPORTED": 0, "UNCERTAIN": 0, "MISATTRIBUTED": 0,
                "FABRICATED": 0, "UNVERIFIABLE": 0,
            }

        rec = CitationRecord(
            case_name=item["case_name"],
            court="Supreme Court",
            year=1990,
            holding=item["holding"],
            self_confidence=0.85,
            agent_source="stress_test",
        )
        rec.reporter_cite = item.get("reporter_cite", "")

        res = cce.compute_ccs(rec)
        tier = res["tier"]
        ccs = res["ccs"]
        accepted = (tier == "SUPPORTED")

        counts = per_type_counts[c_type]
        counts["total"] += 1
        if accepted:
            counts["accepted"] += 1
        else:
            counts["rejected"] += 1

        counts[tier] = counts.get(tier, 0) + 1

        detailed_results.append({
            "type": c_type,
            "case_name": item["case_name"],
            "reporter_cite": item.get("reporter_cite", ""),
            "expected_label": item["label"],
            "verdict": tier,
            "ccs": ccs,
            "existence": res.get("existence", 0.0),
            "support": res.get("support", 0.0),
            "accepted_by_judge": accepted,
        })

    elapsed = time.time() - start_time

    # Key Metrics
    correct_counts = per_type_counts.get("correct", {"total": 0, "accepted": 0})
    power = (correct_counts["accepted"] / correct_counts["total"]) if correct_counts["total"] > 0 else 0.0

    misattributed_counts = per_type_counts.get("misattributed", {"total": 0, "accepted": 0})
    far_misattributed = (misattributed_counts["accepted"] / misattributed_counts["total"]) if misattributed_counts["total"] > 0 else 0.0

    casehold_counts = per_type_counts.get("casehold_distractor", {"total": 0, "accepted": 0})
    far_casehold = (casehold_counts["accepted"] / casehold_counts["total"]) if casehold_counts["total"] > 0 else 0.0

    fab_with_cite = per_type_counts.get("fabricated_with_cite", {"total": 0, "FABRICATED": 0})
    fab_catch_rate = (fab_with_cite.get("FABRICATED", 0) / fab_with_cite["total"]) if fab_with_cite["total"] > 0 else 0.0

    off_corpus = per_type_counts.get("real_off_corpus", {"total": 0, "FABRICATED": 0})
    false_fab_rate = (off_corpus.get("FABRICATED", 0) / off_corpus["total"]) if off_corpus["total"] > 0 else 0.0

    summary_metrics = {
        "total_evaluated": len(dataset),
        "power": round(power, 4),
        "false_acceptance_rate_misattributed": round(far_misattributed, 4),
        "false_acceptance_rate_casehold": round(far_casehold, 4),
        "fabrication_catch_rate": round(fab_catch_rate, 4),
        "false_fabrication_rate": round(false_fab_rate, 4),
        "elapsed_seconds": round(elapsed, 2),
        "per_type_counts": per_type_counts,
        "details": detailed_results,
    }

    return summary_metrics
