ourMemory = "None"
chatLog = ""
BaseKnowledge = """
[Heart Line]
-Starting Position of Heart Line
Starting under the index finger: Content and satisfied with your romantic life.
Starting under the middle finger: Tendency to be selfish or self-centered in love.
Starting in the middle (between fingers): Falls in love easily.
Touching the Life Line: Easily hurt or vulnerable in emotional matters.

-Length & Shape of Heart Line
Short and straight: Shows little interest in romance or sentimentality.
Long and curved: Expresses emotions freely and openly.
Wavy: Frequent romances but few serious or long-term relationships.

-Special Markings of Heart Line
Straight and parallel to the Head Line: Possesses strong emotional control and stability.
Circles (Islands): Indicates periods of sadness or depression.
Broken line: Represents emotional trauma.
Intertwined with small lines: Suggests emotional distress or trauma.

[Head Line]
-Length of Head Line
Short: Prefers physical achievements over intellectual ones.
Clear and long: Deep thinker with a clear mind and excellent focus.

-Position of Head Line
Far from the Life Line: Adventurous nature with a great passion for life.

-Shape of Head Line
Straight: Practical and realistic thinking.
Curved or sloping: Creative, imaginative, and intuitive.
Wavy: Prone to distractions or a short attention span.

-Special Markings of Head Line
Circles or Crosses: Potential for emotional or mental crises.
Broken line: Inconsistency in thought patterns.
Multiple crosses: Indicates a moment for making a life-altering decision.

[Life Line]
-Position of Life Line
Close to the thumb: Tends to tire easily or experiences low energy.
Straight and close to the edge of the palm: Very cautious in social relationships.

-Shape of Life Line
Curved: Abundant energy and vitality.
Sharp semi-circle curve: Strong willpower and intense passion.

-Length of Life Line
Long and deep: High vitality and a robust constitution.
Short and faint: Easily influenced or swayed by others.

-Special Markings of Life Line
Multiple lines: Extraordinary vitality and strength.
Circles (Islands): Possibility of injury or physical illness.
Broken line: A sudden or major change in lifestyle.

[Fate Line]
-Presence of Fate Line
Absent: A "self-made" individual who carves their own path in life.

-Clarity of Fate Line
Deep and clear: Strongly influenced by destiny or external circumstances.

-Special Markings of Fate Line
Breaks or Direction Changes: Major life changes caused by external factors.

-Position of Fate Line
Starts joined with the Life Line: A self-made person who grows with great ambition.
Joins the Life Line in the middle: Situations where one sacrifices their own interests for others.
Starts at the base and crosses the Life Line: Receives significant support from family and friends.
"""

prompt_Purpose = "Your goal is to predict the future through palmistry. Speak in English."
prompt_Memory = f"I will answer by continuing from {chatLog}, the story you just told, with words that do not overlap with {ourMemory}, the summary of the previous conversation."
prompt_Persona = "You are a mysterious fortune teller. You speak in a calm and detached tone."
prompt_RespondFormat = ""
prompt_HowToMakeSentence = "Find cases where the given palmistry characteristics match BaseKnowledge and synthesize them to reconstruct a prophecy predicting the future."

#testing prompts
prompt_HeartLine = "The Heart line meets the life line, and is broken."
prompt_HeadLine = "Short head line, far from life line, straight line"
prompt_LifeLine = "The life line is curved, short, and faint."
prompt_FateLine = "fate line is absent"

