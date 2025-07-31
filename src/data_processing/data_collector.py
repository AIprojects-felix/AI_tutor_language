"""
Data collection module for English-German cross-linguistic learning
"""

import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossLinguisticDataCollector:
    """Cross-linguistic data collector"""
    
    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_sources = {
            'parallel_texts': [],
            'grammar_comparisons': [],
            'vocabulary_pairs': [],
            'common_errors': [],
            'audio_examples': []
        }
        
    def collect_parallel_corpus(self, source_path: Optional[str] = None) -> List[Dict]:
        """Collect English-German parallel corpus"""
        logger.info("Collecting parallel corpus...")
        
        # Sample parallel data
        parallel_data = [
            {
                "id": "parallel_001",
                "english": "I have been learning German for two years.",
                "german": "Ich lerne seit zwei Jahren Deutsch.",
                "alignment": [
                    {"en": "I", "de": "Ich"},
                    {"en": "have been learning", "de": "lerne"},
                    {"en": "German", "de": "Deutsch"},
                    {"en": "for two years", "de": "seit zwei Jahren"}
                ],
                "difficulty": "intermediate",
                "grammar_points": ["present perfect continuous", "present tense with 'seit'"]
            },
            {
                "id": "parallel_002", 
                "english": "The book that I read yesterday was very interesting.",
                "german": "Das Buch, das ich gestern gelesen habe, war sehr interessant.",
                "alignment": [
                    {"en": "The book", "de": "Das Buch"},
                    {"en": "that", "de": "das"},
                    {"en": "I read", "de": "ich gelesen habe"},
                    {"en": "yesterday", "de": "gestern"},
                    {"en": "was very interesting", "de": "war sehr interessant"}
                ],
                "difficulty": "intermediate",
                "grammar_points": ["relative clauses", "past tense"]
            },
            {
                "id": "parallel_003",
                "english": "She will have finished her homework by tomorrow.",
                "german": "Sie wird ihre Hausaufgaben bis morgen fertig haben.",
                "alignment": [
                    {"en": "She", "de": "Sie"},
                    {"en": "will have finished", "de": "wird fertig haben"},
                    {"en": "her homework", "de": "ihre Hausaufgaben"},
                    {"en": "by tomorrow", "de": "bis morgen"}
                ],
                "difficulty": "advanced",
                "grammar_points": ["future perfect", "word order"]
            }
        ]
        
        # Load from external file if provided
        if source_path and os.path.exists(source_path):
            with open(source_path, 'r', encoding='utf-8') as f:
                external_data = json.load(f)
                parallel_data.extend(external_data)
                
        self.data_sources['parallel_texts'] = parallel_data
        logger.info(f"Collected {len(parallel_data)} parallel sentences")
        return parallel_data
    
    def collect_grammar_patterns(self) -> List[Dict]:
        """Collect grammar comparison patterns"""
        logger.info("Collecting grammar patterns...")
        
        grammar_patterns = [
            {
                "id": "grammar_001",
                "topic": "Present Perfect Comparison",
                "english_rule": "have/has + past participle",
                "german_rule": "haben/sein + Partizip II (verb in second position, participle at end)",
                "key_differences": [
                    "German requires choosing the correct auxiliary verb (haben/sein)",
                    "German past participle goes at the end of the sentence",
                    "German uses present tense with 'seit' for ongoing actions"
                ],
                "examples": [
                    {
                        "english": "I have lived here for 5 years.",
                        "german": "Ich wohne seit 5 Jahren hier.",
                        "explanation": "German uses present tense with 'seit' for ongoing actions"
                    },
                    {
                        "english": "She has gone to Berlin.",
                        "german": "Sie ist nach Berlin gefahren.",
                        "explanation": "Motion verbs use 'sein' as auxiliary"
                    }
                ],
                "difficulty": "intermediate",
                "learning_objectives": [
                    "Understand auxiliary verb selection",
                    "Master word order in perfect tenses",
                    "Distinguish between ongoing and completed actions"
                ]
            },
            {
                "id": "grammar_002",
                "topic": "Word Order in Subordinate Clauses",
                "english_rule": "Subject + Verb + Object (same as main clause)",
                "german_rule": "Subject + Object + Verb (verb goes to the end)",
                "key_differences": [
                    "German subordinate clauses have verb-final order",
                    "Separable verbs stay together in subordinate clauses",
                    "Modal verbs create double infinitive constructions"
                ],
                "examples": [
                    {
                        "english": "I know that he speaks German.",
                        "german": "Ich weiß, dass er Deutsch spricht.",
                        "explanation": "Verb 'spricht' moves to the end in subordinate clause"
                    },
                    {
                        "english": "When I get up, I brush my teeth.",
                        "german": "Wenn ich aufstehe, putze ich mir die Zähne.",
                        "explanation": "Separable verb 'aufstehen' stays together"
                    }
                ],
                "difficulty": "intermediate",
                "learning_objectives": [
                    "Master verb placement in subordinate clauses",
                    "Understand separable verb behavior",
                    "Practice complex sentence structures"
                ]
            },
            {
                "id": "grammar_003",
                "topic": "Modal Verbs Comparison",
                "english_rule": "Modal + base form of verb",
                "german_rule": "Modal verb in position 2, main verb infinitive at end",
                "key_differences": [
                    "German modal verbs send main verb to sentence end",
                    "Some German modals have no direct English equivalent",
                    "Past tense formation differs significantly"
                ],
                "examples": [
                    {
                        "english": "I can speak German.",
                        "german": "Ich kann Deutsch sprechen.",
                        "explanation": "Main verb 'sprechen' goes to the end"
                    },
                    {
                        "english": "You should have called me.",
                        "german": "Du hättest mich anrufen sollen.",
                        "explanation": "Complex modal construction with double infinitive"
                    }
                ],
                "difficulty": "advanced",
                "learning_objectives": [
                    "Master modal verb word order",
                    "Understand modal verb meanings",
                    "Practice past tense modal constructions"
                ]
            }
        ]
        
        self.data_sources['grammar_comparisons'] = grammar_patterns
        logger.info(f"Collected {len(grammar_patterns)} grammar patterns")
        return grammar_patterns
    
    def collect_false_friends(self) -> List[Dict]:
        """Collect false friends vocabulary"""
        logger.info("Collecting false friends vocabulary...")
        
        false_friends = [
            {
                "id": "vocab_001",
                "english_word": "become",
                "german_false_friend": "bekommen",
                "english_meaning": "to turn into, to grow to be",
                "german_false_meaning": "to receive, to get",
                "correct_german": "werden",
                "explanation": "This is a classic false friend. 'become' means 'werden' in German, while 'bekommen' means 'to get/receive'.",
                "examples": [
                    {
                        "english": "She became a doctor.",
                        "wrong_german": "Sie bekam eine Ärztin.",
                        "correct_german": "Sie wurde Ärztin.",
                        "explanation": "Use 'werden' for becoming something"
                    }
                ],
                "memory_tips": [
                    "Remember: bekommen = be + come = come to me = receive",
                    "werden sounds like 'were' - think of 'were becoming'"
                ],
                "difficulty": "beginner"
            },
            {
                "id": "vocab_002",
                "english_word": "actual",
                "german_false_friend": "aktuell",
                "english_meaning": "real, true, existing",
                "german_false_meaning": "current, up-to-date",
                "correct_german": "tatsächlich, wirklich",
                "explanation": "'Actual' means 'real/true', while 'aktuell' means 'current/up-to-date'.",
                "examples": [
                    {
                        "english": "The actual cost was higher.",
                        "wrong_german": "Die aktuelle Kosten waren höher.",
                        "correct_german": "Die tatsächlichen Kosten waren höher.",
                        "explanation": "Use 'tatsächlich' for 'actual/real'"
                    }
                ],
                "memory_tips": [
                    "aktuell = actual news = current news",
                    "tatsächlich contains 'Tatsache' (fact)"
                ],
                "difficulty": "intermediate"
            },
            {
                "id": "vocab_003",
                "english_word": "sympathetic",
                "german_false_friend": "sympathisch",
                "english_meaning": "showing compassion, understanding",
                "german_false_meaning": "likeable, nice, pleasant",
                "correct_german": "mitfühlend, verständnisvoll",
                "explanation": "'Sympathetic' means showing compassion, while 'sympathisch' means likeable or nice.",
                "examples": [
                    {
                        "english": "She was sympathetic to my problems.",
                        "wrong_german": "Sie war sympathisch zu meinen Problemen.",
                        "correct_german": "Sie war verständnisvoll gegenüber meinen Problemen.",
                        "explanation": "Use 'verständnisvoll' for showing understanding"
                    }
                ],
                "memory_tips": [
                    "sympathisch = nice person you'd want as friend",
                    "mitfühlend = mit (with) + feeling = feeling with someone"
                ],
                "difficulty": "intermediate"
            }
        ]
        
        self.data_sources['vocabulary_pairs'] = false_friends
        logger.info(f"Collected {len(false_friends)} false friends")
        return false_friends
    
    def collect_common_errors(self) -> List[Dict]:
        """Collect common error patterns"""
        logger.info("Collecting common error patterns...")
        
        common_errors = [
            {
                "id": "error_001",
                "error_type": "Word order with time expressions",
                "level": "intermediate",
                "description": "Verb position error when sentence starts with time expression",
                "example_error": "Heute ich gehe ins Kino.",
                "correct_form": "Heute gehe ich ins Kino.",
                "explanation": "When a sentence starts with a time expression, the verb must come immediately after (V2 principle)",
                "related_rule": "V2 (Verb Second) principle",
                "practice_suggestions": [
                    "Practice making sentences with different adverbials at the start",
                    "Pay attention to subject-verb position changes"
                ]
            },
            {
                "id": "error_002",
                "error_type": "Auxiliary verb selection error",
                "level": "intermediate",
                "description": "Wrong choice between haben/sein in perfect tenses",
                "example_error": "Ich habe nach Berlin gefahren.",
                "correct_form": "Ich bin nach Berlin gefahren.",
                "explanation": "Verbs indicating change of location (like fahren, gehen, kommen) use 'sein' as auxiliary in perfect tenses",
                "related_rule": "Perfekt with sein/haben",
                "practice_suggestions": [
                    "Memorize list of verbs that use 'sein'",
                    "Understand concepts of location change and state change"
                ]
            },
            {
                "id": "error_003",
                "error_type": "Case error after prepositions",
                "level": "advanced",
                "description": "Wrong case after prepositions",
                "example_error": "Ich gehe zu der Schule.",
                "correct_form": "Ich gehe zur Schule.",
                "explanation": "zu + der should be contracted to 'zur'",
                "related_rule": "Preposition + Article contraction",
                "practice_suggestions": [
                    "Memorize common preposition contractions",
                    "Practice preposition + article combinations"
                ]
            },
            {
                "id": "error_004",
                "error_type": "Adjective ending errors",
                "level": "advanced",
                "description": "Wrong adjective endings in different cases",
                "example_error": "Ich sehe einen groß Mann.",
                "correct_form": "Ich sehe einen großen Mann.",
                "explanation": "Adjectives before nouns must agree in case, gender, and number",
                "related_rule": "Adjective declension",
                "practice_suggestions": [
                    "Practice adjective endings with different articles",
                    "Learn the adjective ending patterns systematically"
                ]
            }
        ]
        
        self.data_sources['common_errors'] = common_errors
        logger.info(f"Collected {len(common_errors)} error patterns")
        return common_errors
    
    def save_collected_data(self, output_dir: str = "./data/raw"):
        """Save collected data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for data_type, data in self.data_sources.items():
            if data:
                file_path = output_path / f"{data_type}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {data_type} to {file_path}")
    
    def load_external_data(self, file_path: str, data_type: str) -> List[Dict]:
        """Load external data file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.data_sources[data_type].extend(data)
            logger.info(f"Loaded {len(data)} {data_type} items from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get data statistics"""
        stats = {}
        for data_type, data in self.data_sources.items():
            stats[data_type] = {
                'count': len(data),
                'example': data[0] if data else None
            }
        return stats


if __name__ == "__main__":
    # Test data collector
    collector = CrossLinguisticDataCollector()
    
    # Collect data
    collector.collect_parallel_corpus()
    collector.collect_grammar_patterns()
    collector.collect_false_friends()
    collector.collect_common_errors()
    
    # Save data
    collector.save_collected_data()
    
    # Print statistics
    stats = collector.get_statistics()
    print("Data collection statistics:")
    for dtype, info in stats.items():
        print(f"  {dtype}: {info['count']} items")
