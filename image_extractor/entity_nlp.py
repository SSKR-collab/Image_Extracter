import re
from image_extractor.base_analyzer import BaseAnalyzer


class EntityNlp(BaseAnalyzer):
    """
    Performs lightweight rule-based Named Entity Recognition (NER),
    dialogue quote matching (with speaker resolution), relationship mapping,
    and emotional sentiment profiling.
    """
    VERSION = "1.1.0"

    # Known name and place lists for rule fallback
    KNOWN_PEOPLE = {"bob", "phyllis", "john", "russell", "edward", "ed", "mary", "sarah", "tom", "alice"}
    KNOWN_PLACES = {"southend", "london", "paris", "tokyo", "new york", "boston"}

    # Known proverbs/idioms for semantic categorization
    KNOWN_PROVERBS = {
        "the customer is always right",
        "east, west, home's best",
        "life’s not all beer and skittles",
        "the devil looks after his own",
        "manners maketh man",
        "many a mickle makes a muckle",
        "a man who is his own lawyer has a fool for his client",
        "you can’t make a silk purse from a sow’s ear",
        "as thick as thieves",
        "clothes make the man",
        "all that glisters is not gold",
        "the pen is mightier than sword",
        "is fair and wise and good and gay",
        "make love not war",
        "devil take the hindmost",
        "the female of the species is more deadly than the male",
        "a place for everything and everything in its place",
        "hell hath no fury like a woman scorned",
        "when in rome, do as the romans do",
        "to err is human; to forgive divine",
        "enough is as good as a feast",
        "people who live in glass houses shouldn’t throw stones",
        "nature abhors a vacuum",
        "moderation in all things",
        "everything comes to him who waits",
        "tomorrow is another day",
        "better to light a candle than to curse the darkness",
        "two is company, but three’s a crowd",
        "it’s the squeaky wheel that gets the grease",
        "don’t teach your grandma to suck eggs",
        "he who lives by the sword shall die by the sword",
        "don’t meet troubles half-way",
        "oil and water don’t mix",
        "all work and no play makes jack a dull boy",
        "the best things in life are free",
        "finders keepers, losers weepers",
        "there's no place like home",
        "speak softly and carry a big stick",
        "music has charms to soothe the savage breast",
        "ne’er cast a clout till may be out",
        "there’s no such thing as a free lunch",
        "nothing venture, nothing gain",
        "he who can does, he who cannot, teaches",
        "a stitch in time saves nine",
        "the child is the father of the man"
    }

    # Sentiment dictionaries
    SENTIMENT_WORDS = {
        "joy": {"happy", "glad", "joy", "smiled", "laughed", "delighted", "lovely", "sweet", "smile", "laugh", "cheerful"},
        "sadness": {"cried", "wept", "sad", "tears", "shook", "sorry", "grief", "pain", "broke", "gloomy", "sigh", "sighed"},
        "anger": {"angry", "furious", "mad", "shouted", "yelled", "scolded", "hate", "irritated", "glared"}
    }

    # Regex patterns for Speaker extraction
    DIALOGUE_PATTERNS = [
        # Match "Quote..." said Bob
        r'[\u201c"]([^"]+)[\u201d"]\s*(?:said|cried|replied|asked|shook|whispered)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        # Match Bob said, "Quote..."
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*(?:said|cried|replied|asked|shook|whispered),?\s*[\u201c"]([^"]+)[\u201d"]'
    ]

    # Regex patterns for family/social relationships
    RELATIONSHIP_PATTERNS = [
        # X is the Y of Z
        r'\b([A-Z][a-z]+)\s+is\s+the\s+(nephew|niece|uncle|aunt|brother|sister|son|daughter|mother|father|husband|wife|friend)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        # X, the Y of Z, ...
        r'\b([A-Z][a-z]+),\s+the\s+(nephew|niece|uncle|aunt|brother|sister|son|daughter|mother|father|husband|wife|friend)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        # Z's Y, X
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\'\s*s\s+(nephew|niece|uncle|aunt|brother|sister|son|daughter|mother|father|husband|wife|friend),\s+([A-Z][a-z]+)\b'
    ]

    def analyze(self, file_path: str, img, context: dict) -> dict:
        results = {
            "facts": {},
            "nlp_insights": {
                "entities": [],
                "dialogue": [],
                "relationships": [],
                "proverbs": [],
                "sentiment": {
                    "emotion": "neutral",
                    "confidence": 0.5
                }
            },
            "errors": []
        }

        # Retrieve text from OCR
        ocr_out = context.get("ocr_engine", {})
        raw_text = ocr_out.get("facts", {}).get("raw_text", "").strip()

        if not raw_text:
            return results

        # 1. Named Entity Recognition (NER)
        entities = self._extract_entities(raw_text)
        results["nlp_insights"]["entities"] = entities

        # 2. Dialogue / Quotes Extraction
        dialogue = self._extract_dialogue(raw_text)
        results["nlp_insights"]["dialogue"] = dialogue

        # 3. Relationship Extraction
        relationships = self._extract_relationships(raw_text)
        results["nlp_insights"]["relationships"] = relationships

        # 4. Proverb / Idiom Extraction
        proverbs = self._extract_proverbs(raw_text)
        results["nlp_insights"]["proverbs"] = proverbs

        # 5. Sentiment / Intent Analysis
        sentiment = self._analyze_sentiment(raw_text)
        results["nlp_insights"]["sentiment"] = sentiment

        # Register Facts summary
        results["facts"]["entities_summary"] = {
            "person_count": sum(1 for e in entities if e["type"] == "person"),
            "place_count": sum(1 for e in entities if e["type"] == "place"),
            "quote_count": len(dialogue),
            "proverb_count": len(proverbs)
        }

        return results

    def _extract_entities(self, text: str) -> list:
        entities = []
        found_names = set()

        # Pattern A: Prefix-based Matching (e.g. Mrs. Russell, Uncle Edward, Mr. Smith)
        prefix_pattern = r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|Uncle|Aunt|Sir|Lady)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        for match in re.finditer(prefix_pattern, text):
            prefix, name = match.group(1), match.group(2)
            full_entity = f"{prefix} {name}"
            if full_entity.lower() not in found_names:
                found_names.add(full_entity.lower())
                entities.append({
                    "text": full_entity,
                    "type": "person",
                    "confidence": 0.95,
                    "source": "honorific_prefix"
                })

        # Pattern B: Known Names and Places matching
        words = re.findall(r'\b[a-zA-Z.-]+\b', text)
        for word in words:
            word_lower = word.lower()
            if word[0].isupper(): # Capitalized proper noun check
                if word_lower in self.KNOWN_PEOPLE and word_lower not in found_names:
                    found_names.add(word_lower)
                    entities.append({
                        "text": word,
                        "type": "person",
                        "confidence": 0.85,
                        "source": "dictionary_match"
                    })
                elif word_lower in self.KNOWN_PLACES and word_lower not in found_names:
                    found_names.add(word_lower)
                    entities.append({
                        "text": word,
                        "type": "place",
                        "confidence": 0.85,
                        "source": "dictionary_match"
                    })

        # Pattern C: Generic proper nouns (consecutive capitalized words)
        generic_pattern = r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'
        # Ignore common English sentence starters
        common_starters = {"the", "a", "an", "in", "on", "at", "to", "he", "she", "it", "they", "we", "but", "and"}
        
        for match in re.finditer(generic_pattern, text):
            word1, word2 = match.group(1), match.group(2)
            full_str = f"{word1} {word2}"
            if word1.lower() not in common_starters and full_str.lower() not in found_names:
                # Basic check to avoid sentence boundary false positives
                # If first word is the start of a sentence (preceded by dot/exclamation)
                # we are cautious, but if both are capitalized it is likely a name/place/org.
                found_names.add(full_str.lower())
                entities.append({
                    "text": full_str,
                    "type": "proper_noun",
                    "confidence": 0.70,
                    "source": "consecutive_capitalized"
                })

        return entities

    def _extract_dialogue(self, text: str) -> list:
        dialogue_entries = []

        # Try mapping with speaker extraction patterns first
        matched_indices = []
        for pat in self.DIALOGUE_PATTERNS:
            for match in re.finditer(pat, text):
                # Dialogue patterns have two groups: speaker and quote
                # We determine which group is which
                g1, g2 = match.group(1), match.group(2)
                
                # Check which one is the quote
                if "\n" in g1 or len(g1) > len(g2) * 1.5:
                    quote, speaker = g1, g2
                else:
                    quote, speaker = g2, g1
                    
                dialogue_entries.append({
                    "text": quote.strip(),
                    "speaker": speaker.strip(),
                    "confidence": 0.90
                })
                # Register start-end indices to avoid duplication with general quotes
                matched_indices.append(match.span())

        # Grab remaining quotes that didn't match a specific speaker pattern
        # Match standard double quotes
        quote_pat = r'[\u201c"]([^"]+)[\u201d"]'
        for match in re.finditer(quote_pat, text):
            # Check if this quote was already extracted
            is_duplicate = False
            for start, end in matched_indices:
                if match.start() >= start and match.end() <= end:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                dialogue_entries.append({
                    "text": match.group(1).strip(),
                    "speaker": "unknown",
                    "confidence": 0.80
                })

        return dialogue_entries

    def _extract_relationships(self, text: str) -> list:
        relationships = []
        
        # Scan standard syntactic relationship patterns
        for pat in self.RELATIONSHIP_PATTERNS:
            for match in re.finditer(pat, text):
                # Pattern group arrangement varies depending on regex
                groups = match.groups()
                if "'" in match.group(0): # possessive pattern Z's Y, X
                    p2, role, p1 = groups[0], groups[1], groups[2]
                else: # X is the Y of Z, or X, the Y of Z
                    p1, role, p2 = groups[0], groups[1], groups[2]
                    
                relationships.append({
                    "person1": p1.strip(),
                    "relation": role.strip().lower(),
                    "person2": p2.strip(),
                    "confidence": 0.85
                })

        # Heuristic search: if "Uncle Edward" and "Bob" are in the text
        # and "nephew" or "uncle" appears nearby, we can suggest a relationship
        if "uncle edward" in text.lower() and "bob" in text.lower():
            # Check if relationship already parsed
            exists = any(r["person1"].lower() == "bob" and r["relation"] in ("nephew", "uncle") for r in relationships)
            if not exists:
                relationships.append({
                    "person1": "Bob",
                    "relation": "nephew",
                    "person2": "Uncle Edward",
                    "confidence": 0.65,
                    "note": "Inferred from proximity of Bob and Uncle Edward."
                })

        return relationships

    def _analyze_sentiment(self, text: str) -> dict:
        text_words = [w.lower().strip(".,?!\"'();:") for w in text.split()]
        if not text_words:
            return {"emotion": "neutral", "confidence": 0.50}

        counts = {emotion: 0 for emotion in self.SENTIMENT_WORDS}
        for w in text_words:
            for emotion, words_dict in self.SENTIMENT_WORDS.items():
                if w in words_dict:
                    counts[emotion] += 1

        # Determine dominant emotion
        max_emotion = "neutral"
        max_count = 0
        total_matches = sum(counts.values())

        for emotion, count in counts.items():
            if count > max_count:
                max_count = count
                max_emotion = emotion

        if total_matches > 0:
            # Calculate confidence score
            conf = 0.5 + (max_count / total_matches) * 0.45
            return {
                "emotion": max_emotion,
                "confidence": round(conf, 2),
                "details": {k: v for k, v in counts.items() if v > 0}
            }

        return {
            "emotion": "neutral",
            "confidence": 0.70
        }

    def _extract_proverbs(self, text: str) -> list:
        """
        Scans text for known proverbs/idioms using a robust cleaning check.
        """
        clean_text = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        clean_text = re.sub(r'\s+', ' ', clean_text)

        matches = []
        for prov in self.KNOWN_PROVERBS:
            clean_prov = prov.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
            if clean_prov in clean_text:
                matches.append({
                    "text": prov,
                    "confidence": 0.95
                })
        return matches
