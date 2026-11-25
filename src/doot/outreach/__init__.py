"""Contact discovery and grounded cold outreach package."""

from doot.outreach.emailer import ColdEmailDraft, draft_cold_email, send_outreach_email
from doot.outreach.finder import (
    ContactDiscoveryResult,
    check_domain_has_mx,
    extract_emails_from_text,
    find_contacts_for_company,
    generate_email_candidates,
    infer_company_domain,
    parse_linkedin_people_hits,
)

__all__ = [
    "ColdEmailDraft",
    "ContactDiscoveryResult",
    "check_domain_has_mx",
    "draft_cold_email",
    "extract_emails_from_text",
    "find_contacts_for_company",
    "generate_email_candidates",
    "infer_company_domain",
    "parse_linkedin_people_hits",
    "send_outreach_email",
]
