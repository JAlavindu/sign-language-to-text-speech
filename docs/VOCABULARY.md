# 20 Common Sign Vocabulary for MVP

## Selected Signs (American Sign Language - ASL)

These 20 signs are chosen for:

- High frequency in daily conversation
- Distinct sensor patterns (easy to differentiate)
- Mix of static poses and dynamic movements
- Practical utility for basic communication

### Category: Greetings & Polite Phrases (5)

1. **Hello** - Open hand wave
2. **Goodbye** - Palm out, fingers open/close
3. **Thank you** - Fingers touch chin, move forward
4. **Please** - Circular motion on chest
5. **Sorry** - Fist circles on chest

### Category: Common Responses (5)

6. **Yes** - Fist nods like head nodding
7. **No** - Index and middle finger tap thumb (like snapping)
8. **OK/Fine** - Fingerspell O-K
9. **Good** - Hand moves from chin outward
10. **Bad** - Hand moves from chin, rotates down

### Category: Questions (4)

11. **What** - Index finger sweeps across open palm
12. **Who** - Index finger circles near mouth
13. **Where** - Index finger shakes side to side
14. **Help** - Thumbs up, one hand lifts the other

### Category: Essential Needs (3)

15. **Eat/Food** - Fingers tap mouth
16. **Drink/Water** - C-shape hand to mouth (like holding cup)
17. **Stop** - Flat hand chops down onto open palm

### Category: Time & Basic Concepts (3)

18. **Now** - Both hands move down (using one hand for glove)
19. **Later** - L-shape hand, index finger moves forward
20. **Understand** - Index finger flicks up from forehead

---

## Sign Characteristics for Sensor Fusion

### Static Signs (Pose-dominant)

- OK, Yes, Good, Bad, Later
- **Primary sensors:** Flex sensors (finger positions)
- **Secondary:** IMU (hand orientation)

### Dynamic Signs (Movement-dominant)

- Hello, Goodbye, Thank you, Please, Sorry, What, Where
- **Primary sensors:** IMU (acceleration patterns, angular velocity)
- **Secondary:** Flex sensors (maintain finger configuration during motion)

### Contact Signs (Finger-to-body or finger-to-finger)

- Thank you, Please, Sorry, Good, Bad, Eat, Drink, Who
- **Primary sensors:** Capacitive touch (detecting contact)
- **Secondary:** IMU + Flex (position context)

### Complex Signs (Multiple phases)

- Help (two hands - we'll use one-hand variation)
- Stop (two hands - we'll use dominant hand motion)

---

## Data Collection Strategy

For each sign, record:

- **10 repetitions per person**
- **3-5 different speeds** (slow, normal, fast)
- **Multiple starting positions** (hand at side, in front, different heights)
- **Include transitions** (how you move between signs)

Minimum dataset size:

- 5 people × 20 signs × 10 reps = **1000 samples**
- Add variations = ~**1500-2000 samples** total

---

## Expansion Path (Future Vocabulary)

After MVP works, add:

- Alphabet fingerspelling (A-Z)
- Numbers (0-9)
- Common nouns (family, work, home, etc.)
- Verbs (go, come, see, know, etc.)
- Adjectives (big, small, hot, cold, etc.)

Target: **100+ signs** for basic conversational ability

---

## ASL Resources for Learning Signs

- **Lifeprint.com** - Comprehensive ASL dictionary with videos
- **HandSpeak.com** - ASL dictionary and learning resources
- **SignASL.org** - Video demonstrations
- **YouTube channels:** Bill Vicars (ASL University), Sign Duo

**Important:** Get feedback from actual ASL users to ensure your signs are accurate and respectful of Deaf culture!
