"""
Meeting Task Extraction Service using spaCy NER + Rule-Based Detection.

This service extracts tasks from meeting transcriptions by:
1. Using spaCy NER to extract PERSON, DATE, TIME entities
2. Using rule-based pattern matching to detect task trigger verbs
3. Combining entities and patterns to structure tasks with assignees and due dates
"""
import re
from typing import List, Dict, Optional, Tuple

# --- spaCy disabled (not installed; meeting task extraction not in use) ---
# try:
#     import spacy
#     from spacy.matcher import Matcher
#     SPACY_AVAILABLE = True
# except ImportError:
#     SPACY_AVAILABLE = False
#     print("Warning: spaCy not installed. Meeting task extraction will not work.")
SPACY_AVAILABLE = False


# Task trigger verbs (imperative/modal verbs that indicate tasks)
# Excluded: "show", "see", "come", "have", "do", "go", "sit", "take", "remain", "bring"
# These are too common in descriptive speech and not actionable tasks
TASK_TRIGGER_VERBS = {
    "send", "deliver", "prepare", "review", "schedule", "update", "finalize",
    "share", "call", "follow up", "follow-up", "complete", "finish", "submit",
    "create", "write", "draft", "build", "implement", "fix", "resolve",
    "check", "verify", "confirm", "arrange", "organize", "coordinate",
    "meet", "discuss", "present", "provide", "give", "ensure", "address",
    "consider", "approve", "reject", "decide", "determine", "evaluate",
    "investigate", "analyze", "develop", "design", "plan", "propose"
}

# Verbs that are NOT tasks (descriptive/passive verbs)
NON_TASK_VERBS = {
    "show", "see", "come", "have", "do", "go", "sit", "take", "remain",
    "bring", "look", "watch", "view", "display", "indicate", "demonstrate",
    "appear", "seem", "be", "is", "are", "was", "were", "been", "being"
}

# Modal verbs that indicate tasks
MODAL_VERBS = {"should", "will", "need to", "must", "have to", "ought to"}


class MeetingTaskExtractor:
    """Extract tasks from meeting transcriptions using spaCy NER + rule-based patterns."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the task extractor with a spaCy model.
        
        Args:
            model_name: spaCy model name (e.g., "en_core_web_sm" or "en_core_web_trf")
        """
        if not SPACY_AVAILABLE:
            raise ImportError(
                "spaCy is not installed. Please install it with: "
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
        
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # Try to download the model if not found
            import subprocess
            import sys
            print(f"Model {model_name} not found. Attempting to download...")
            subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=False)
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                # Fallback to small model
                print(f"Failed to load {model_name}, falling back to en_core_web_sm")
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    raise ImportError(
                        "spaCy model not found. Please run: "
                        "python -m spacy download en_core_web_sm"
                    )
        
        # Initialize spaCy Matcher for pattern matching
        self.matcher = Matcher(self.nlp.vocab)
        self._add_task_patterns()
    
    def _add_task_patterns(self):
        """Add patterns to the matcher for detecting task sentences."""
        # Pattern 1: [PERSON] + [imperative verb] + [object]
        # Example: "John, please send the proposal"
        pattern1 = [
            {"ENT_TYPE": "PERSON"},  # Person name entity
            {"LOWER": {"IN": list(TASK_TRIGGER_VERBS)}},
        ]
        
        # Pattern 2: [imperative verb] + [object] + [PERSON]
        # Example: "Send the proposal to John"
        pattern2 = [
            {"LOWER": {"IN": list(TASK_TRIGGER_VERBS)}},
            {"ENT_TYPE": "PERSON", "OP": "?"},
        ]
        
        # Pattern 3: [modal verb] + [verb] + [object] + [PERSON]
        # Example: "John should send the proposal"
        pattern3 = [
            {"LOWER": {"IN": list(MODAL_VERBS)}},
            {"POS": "VERB", "OP": "?"},
        ]
        
        self.matcher.add("TASK_PATTERN", [pattern1, pattern2, pattern3])
    
    def _is_valid_date(self, date_text: str) -> bool:
        """Check if a date string is a valid, meaningful date (not relative time)."""
        if not date_text:
            return False
        
        date_lower = date_text.lower()
        
        # Filter out timestamps
        if re.match(r'^\d+:\d+:\d+', date_lower):
            return False
        
        # Filter out relative time expressions that are too vague
        invalid_patterns = [
            r'^a\s+(minute|hour)',  # "a minute", "an hour" (too short-term)
            r'^some\s+',  # "some six years" (too vague)
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, date_lower):
                return False
        
        # Accept specific date indicators
        date_indicators = ['march', 'april', 'may', 'june', 'july', 'august', 
                          'september', 'october', 'november', 'december',
                          'january', 'february', 'monday', 'tuesday', 'wednesday',
                          'thursday', 'friday', 'saturday', 'sunday',
                          'st', 'nd', 'rd', 'th', '2016', '2017', '2018', '2019',
                          '2020', '2021', '2022', '2023', '2024', '2025', '2026']
        
        if any(indicator in date_lower for indicator in date_indicators):
            return True
        
        # Accept "tonight", "this evening", "tomorrow", "next week" if they're part of a task
        # (these are acceptable as due dates in meeting contexts)
        acceptable_relative = ['tonight', 'this evening', 'this morning', 
                              'this afternoon', 'tomorrow', 'next week', 
                              'next month', 'friday', 'monday', 'tuesday',
                              'wednesday', 'thursday', 'saturday', 'sunday']
        
        if date_lower in acceptable_relative:
            return True
        
        # If it's a very short phrase (1-2 words) without date indicators, reject
        words = date_text.split()
        if len(words) <= 2 and not any(indicator in date_lower for indicator in date_indicators):
            if date_lower not in acceptable_relative:
                return False
        
        return True
    
    def _is_valid_person(self, person_text: str) -> bool:
        """Check if a person string is valid (not a timestamp or other entity)."""
        if not person_text:
            return False
        
        # Filter out timestamps
        if re.match(r'^\d{2}:\d{2}:\d{2}', person_text):
            return False
        
        # Filter out very short or numeric strings
        if len(person_text.strip()) < 2 or person_text.strip().isdigit():
            return False
        
        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', person_text):
            return False
        
        return True
    
    def _calculate_confidence(self, has_person: bool, has_date: bool, has_verb: bool, 
                             action_phrase_length: int, has_modal: bool) -> float:
        """
        Calculate confidence score for a detected task.
        
        Confidence heuristic:
        - Action verb present: +0.4 (base requirement)
        - Modal verb present: +0.2 (stronger indicator)
        - PERSON present: +0.2
        - Valid DATE present: +0.2
        - Action phrase length (longer = more specific): +0.1 if > 3 words
        """
        confidence = 0.0
        
        if has_verb:
            confidence += 0.4
        if has_modal:
            confidence += 0.2
        if has_person:
            confidence += 0.2
        if has_date:
            confidence += 0.2
        if action_phrase_length > 3:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_action_phrase(self, sent, verb_idx: int) -> str:
        """
        Extract a complete, contextual action phrase from a sentence.
        Includes verb, object, and relevant context to make the task clear.
        """
        verb_token = sent[verb_idx]
        
        # Strategy: Extract from verb to end of sentence, but intelligently
        # Include verb + all its dependents + prepositional phrases
        
        # Find the verb's subtree (all tokens that depend on this verb)
        verb_subtree = set([verb_token.i])
        for token in sent:
            if token.head.i == verb_token.i:
                for child in token.subtree:
                    verb_subtree.add(child.i)
        
        # Collect tokens from verb onwards, but stop at:
        # - Another main verb (different clause)
        # - Sentence-ending punctuation
        # - Too far from verb (max 15 words)
        action_tokens = []
        start_idx = verb_idx
        max_tokens = 15  # Reasonable limit for task description
        
        # Find where the verb phrase ends
        # Look for the next main verb or clause boundary
        end_idx = len(sent)
        for i in range(verb_idx + 1, len(sent)):
            token = sent[i]
            # Stop if we hit another main verb (different clause)
            if token.pos_ == "VERB" and token.dep_ == "ROOT" and token.i != verb_token.i:
                end_idx = i
                break
            # Stop at coordinating conjunctions that start new clauses
            if token.text.lower() in ["and", "but", "or"] and i > verb_idx + 3:
                # Check if this starts a new independent clause
                if any(t.pos_ == "VERB" for t in sent[i+1:min(i+5, len(sent))]):
                    end_idx = i
                    break
        
        # Collect tokens from verb to end_idx - be more inclusive for context
        collected = set()
        for i in range(start_idx, min(end_idx, start_idx + max_tokens)):
            token = sent[i]
            
            # Skip if it's a person/date entity (we'll add those separately)
            if any(ent.start <= token.i < ent.end for ent in sent.ents 
                   if ent.label_ in ["PERSON", "DATE", "TIME"]):
                continue
            
            # Include the token if it's part of the verb phrase or relevant
            # Be more inclusive - include most tokens after the verb for context
            if (token.i in verb_subtree or 
                token.head.i in verb_subtree or
                token.dep_ in ["dobj", "pobj", "attr", "prep", "advmod", "amod", "nmod", "det", "ccomp", "xcomp"] or
                i < start_idx + 12):  # Include tokens within 12 positions for context
                
                # Include articles and determiners for natural phrasing
                if token.i not in collected:
                    action_tokens.append(token)
                    collected.add(token.i)
        
        # If we didn't get enough context (less than 5 words), include more following tokens
        if len(action_tokens) < 5:
            for i in range(start_idx, min(start_idx + 15, len(sent))):
                token = sent[i]
                if token.i not in collected:
                    # Skip entities but include other words for context
                    if not any(ent.start <= token.i < ent.end for ent in sent.ents 
                               if ent.label_ in ["PERSON", "DATE", "TIME"]):
                        action_tokens.append(token)
                        collected.add(token.i)
                        if len(action_tokens) >= 10:  # Get at least 10 words for good context
                            break
        
        # Build the phrase from collected tokens
        if not action_tokens:
            # Fallback: just the verb
            return verb_token.text
        
        # Get text in order
        phrase_parts = [t.text for t in action_tokens]
        result = " ".join(phrase_parts)
        
        # Clean up: normalize spaces, remove extra punctuation
        result = re.sub(r'\s+', ' ', result)
        # Remove leading/trailing punctuation but keep internal punctuation
        result = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', result)
        
        # Ensure we have at least the verb
        if not result or len(result.split()) < 1:
            result = verb_token.text
        
        return result.strip()
    
    def _find_task_verb(self, sent) -> Tuple[Optional[int], bool]:
        """
        Find the index of a task trigger verb in the sentence.
        Returns (verb_index, has_modal) tuple.
        """
        has_modal = False
        modal_idx = None
        
        # First, check for modal verbs
        for i, token in enumerate(sent):
            if token.lemma_.lower() in MODAL_VERBS:
                has_modal = True
                modal_idx = i
                # Check if next token is a task verb
                if i + 1 < len(sent) and sent[i + 1].lemma_.lower() in TASK_TRIGGER_VERBS:
                    return (i + 1, True)
                # Check if next token is any verb
                if i + 1 < len(sent) and sent[i + 1].pos_ == "VERB":
                    verb_lemma = sent[i + 1].lemma_.lower()
                    if verb_lemma not in NON_TASK_VERBS:
                        return (i + 1, True)
        
        # Look for task trigger verbs directly
        for i, token in enumerate(sent):
            if token.lemma_.lower() in TASK_TRIGGER_VERBS:
                # If we found a modal earlier, prefer verbs after modals
                if has_modal and modal_idx is not None and i > modal_idx:
                    return (i, True)
                return (i, has_modal)
        
        return (None, False)
    
    def extract_tasks(self, transcription: str) -> List[Dict]:
        """
        Extract tasks from a meeting transcription.
        
        Args:
            transcription: Full meeting transcription text
            
        Returns:
            List of task dictionaries with structure:
            {
                "task": str,           # Action phrase
                "assignee": str | None, # Person assigned
                "due_date": str | None, # Date mentioned
                "confidence": float     # Confidence score (0.0-1.0)
            }
        """
        if not transcription or not transcription.strip():
            return []
        
        # Split transcription into sentences
        doc = self.nlp(transcription)
        sentences = [sent for sent in doc.sents]
        
        tasks = []
        
        for sent in sentences:
            # Skip very short sentences (less than 15 chars)
            if len(sent.text.strip()) < 15:
                continue
            
            # Extract and filter entities
            persons = [ent.text for ent in sent.ents 
                      if ent.label_ == "PERSON" and self._is_valid_person(ent.text)]
            dates = [ent.text for ent in sent.ents 
                    if ent.label_ in ["DATE", "TIME"] and self._is_valid_date(ent.text)]
            
            # Find task trigger verbs
            verb_idx, has_modal = self._find_task_verb(sent)
            
            # Must have a task verb to be considered
            if verb_idx is None:
                continue
            
            # Check if this sentence matches task patterns
            matches = self.matcher(sent)
            
            # Task candidate criteria:
            # 1. Matches pattern, OR
            # 2. Has task verb + (person OR valid date), OR
            # 3. Has modal verb + task verb (strong indicator)
            is_task_candidate = (
                len(matches) > 0 or
                (verb_idx is not None and (len(persons) > 0 or len(dates) > 0)) or
                (has_modal and verb_idx is not None)
            )
            
            if not is_task_candidate:
                continue
            
            # Extract action phrase
            action_phrase = self._extract_action_phrase(sent, verb_idx)
            
            # Filter out very short or meaningless phrases (need at least 3 words for context)
            if not action_phrase or len(action_phrase.split()) < 3:
                continue
            
            # Filter out phrases that are just common non-task verbs
            first_word = action_phrase.split()[0].lower()
            if first_word in NON_TASK_VERBS:
                continue
            
            # Filter out incomplete phrases that end with prepositions or articles
            last_word = action_phrase.split()[-1].lower()
            if last_word in ["the", "a", "an", "of", "to", "for", "in", "on", "at", "by", "with"]:
                # Try to extend the phrase by including more tokens from the sentence
                # Find all tokens from verb to end of sentence (or next clause)
                extended_tokens = []
                collected_ids = set()
                
                # Start from verb and collect up to 12 more tokens
                for i in range(verb_idx, min(verb_idx + 15, len(sent))):
                    token = sent[i]
                    # Skip if it's a person/date entity (already captured separately)
                    is_entity = any(ent.start <= token.i < ent.end for ent in sent.ents 
                                   if ent.label_ in ["PERSON", "DATE", "TIME"])
                    if not is_entity and token.i not in collected_ids:
                        extended_tokens.append(token.text)
                        collected_ids.add(token.i)
                        # Stop if we hit sentence-ending punctuation or another main verb
                        if token.text in [".", "!", "?", ";"]:
                            break
                        if i > verb_idx and token.pos_ == "VERB" and token.dep_ == "ROOT":
                            break
                
                if len(extended_tokens) > len(action_phrase.split()):
                    extended_phrase = " ".join(extended_tokens)
                    # Clean up trailing incomplete words
                    extended_phrase = re.sub(r'\s+(the|a|an|of|to|for|in|on|at|by|with)$', '', extended_phrase)
                    if len(extended_phrase.split()) >= 3:
                        action_phrase = extended_phrase
                    else:
                        continue  # Skip if we can't get a complete phrase
                else:
                    continue  # Skip incomplete phrases
            
            # Extract assignee (first valid person mentioned, or None)
            assignee = persons[0] if persons else None
            
            # Extract due date (first valid date mentioned, or None)
            due_date = dates[0] if dates else None
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                has_person=len(persons) > 0,
                has_date=len(dates) > 0,
                has_verb=verb_idx is not None,
                action_phrase_length=len(action_phrase.split()),
                has_modal=has_modal
            )
            
            # Only include tasks with higher confidence (0.6 minimum)
            # This filters out weak matches
            if confidence >= 0.6:
                tasks.append({
                    "task": action_phrase,
                    "assignee": assignee,
                    "due_date": due_date,
                    "confidence": round(confidence, 2)
                })
        
        # Remove duplicates (same task + assignee combination)
        seen = set()
        unique_tasks = []
        for task in tasks:
            key = (task["task"].lower(), task["assignee"])
            if key not in seen:
                seen.add(key)
                unique_tasks.append(task)
        
        return unique_tasks


def extract_meeting_tasks(transcription: str, model_name: str = "en_core_web_sm") -> Dict:
    """
    Extract tasks from a meeting transcription.
    
    Args:
        transcription: Full meeting transcription text
        model_name: spaCy model name (default: "en_core_web_sm")
    
    Returns:
        Dictionary with structure:
        {
            "tasks": [
                {
                    "task": str,
                    "assignee": str | None,
                    "due_date": str | None,
                    "confidence": float
                },
                ...
            ]
        }
    """
    if not SPACY_AVAILABLE:
        return {
            "tasks": [],
            "error": "spaCy is not installed. Please install it with: pip install spacy && python -m spacy download en_core_web_sm"
        }
    
    try:
        extractor = MeetingTaskExtractor(model_name=model_name)
        tasks = extractor.extract_tasks(transcription)
        
        return {
            "tasks": tasks
        }
    except Exception as e:
        return {
            "tasks": [],
            "error": f"Error extracting tasks: {str(e)}"
        }
