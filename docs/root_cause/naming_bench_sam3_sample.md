| model | sightings | named right | wrong name | not named | false finds per picture | s per picture |
|---|---|---|---|---|---|---|
| vlm (saved answers; its frames only) | 11 | 91% | 9% | 0% | 0.8 | 0.00 |
| vlm (saved answers; its frames only), aliases | 11 | 100% | 0% | 0% | 0.8 | 0.00 |
| owl (as the explorer uses it) | 113 | 42% | 36% | 21% | 1.6 | 0.80 |
| owl (as the explorer uses it), aliases | 113 | 50% | 28% | 21% | 1.6 | 0.00 |
| hybrid: owl, siglip under 1000 px | 113 | 59% | 18% | 23% | 1.6 | 0.85 |
| hybrid: owl, siglip under 1000 px, aliases | 113 | 65% | 12% | 23% | 1.6 | 0.00 |
| sam3 picks among siglip top 3 for owl boxes | 113 | 59% | 17% | 24% | 1.1 | 19.65 |
| sam3 picks among siglip top 3 for owl boxes, aliases | 113 | 71% | 5% | 24% | 1.1 | 0.00 |

Named right, by visible pixels:

| model | 0-100 | 100-300 | 300-1000 | 1000-3000 | 3000-more |
|---|---|---|---|---|---|
| vlm (saved answers; its frames only) |  | 100% (2) | 83% (6) | 100% (3) |  |
| vlm (saved answers; its frames only), aliases |  | 100% (2) | 100% (6) | 100% (3) |  |
| owl (as the explorer uses it) | 15% (33) | 38% (29) | 47% (30) | 83% (18) | 100% (3) |
| owl (as the explorer uses it), aliases | 21% (33) | 48% (29) | 57% (30) | 89% (18) | 100% (3) |
| hybrid: owl, siglip under 1000 px | 39% (33) | 59% (29) | 63% (30) | 83% (18) | 100% (3) |
| hybrid: owl, siglip under 1000 px, aliases | 42% (33) | 66% (29) | 73% (30) | 89% (18) | 100% (3) |
| sam3 picks among siglip top 3 for owl boxes | 36% (33) | 59% (29) | 63% (30) | 89% (18) | 100% (3) |
| sam3 picks among siglip top 3 for owl boxes, aliases | 42% (33) | 76% (29) | 80% (30) | 94% (18) | 100% (3) |

Most frequent wrong names:

- vlm (saved answers; its frames only): cup as mug (1)
- vlm (saved answers; its frames only), aliases: none
- owl (as the explorer uses it): cup as mug (5), bottled water as condiment bottle (5), lemon as pear (4), mug as cup (4), lemon as avocado (4), bottled water as spray (3)
- owl (as the explorer uses it), aliases: bottled water as condiment bottle (5), lemon as pear (4), lemon as avocado (4), bottled water as spray (3), banana as sponge (2), spoon as spatula (2)
- hybrid: owl, siglip under 1000 px: lemon as lime (4), cup as mug (2), mug as burrito (2), bowl as cutting board (1), spoon as banana (1), apple as cherry (1)
- hybrid: owl, siglip under 1000 px, aliases: mug as burrito (2), bowl as cutting board (1), spoon as banana (1), apple as cherry (1), apple as egg (1), lemon as avocado (1)
- sam3 picks among siglip top 3 for owl boxes: mug as cup (11), pan as saucepan (2), bowl as saucepan (1), apple as cherry (1), cup as bowl (1), lemon as apple (1)
- sam3 picks among siglip top 3 for owl boxes, aliases: bowl as saucepan (1), apple as cherry (1), cup as bowl (1), lemon as apple (1), lemon as banana (1), bowl as tray (1)
