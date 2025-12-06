Hey strategists! Some of you may have heard that I've been tinkering with a metric for evaluating board position that I call Material Score (analogous to Chess' use of the same term). The basic idea is to score each player's hand + board in a way that can be used to identify who's ahead. Roughly, it calculates how many turns it would take a player to reach an equivalent state if they lost all the cards in their hand + on their field. Each player gets:
1 material per card in hand
2 material per card on their field (excluding jacks)
1/2 material if it's their turn

Intuitively, I think material captures board position fairly well because any card in your hand would take you to one turn to draw, and any card on your field would take you one turn to draw, plus an additional turn to play. And all other things being equal, it's better if it's your turn.

That said, my instinct is also that there should be some caveats. I'd love your feedback!

2+ glasses should only count as one card on the board (they add no extra benefit)
3+ queens should count only as two cards on the board (3rd & 4th queen add no benefit)
Points and kings are actually fairly nuanced in their value. Having 2 point cards that sum to <= 10 is enormously weaker than having two point cards which sum to >= 11. Similarly King & A-3 is not nearly as strong as King & 4+. Basically I think that we should ignore 'extra' offensive cards on the board which don't get you a turn closer to check. This is the messiest rule but one that looks to me like it would help to differentiate some very poor moves from stronger ones (e.g. King + 3 from King + 4)


What do you think? It's my hope to release an official public data set on cuttle gameplay that the community can use as a basis for strategic analysis. One of the questions that I have is whether to include material scores "out of the box" or leave it to the community to compute it. Penny for your thoughts! 
