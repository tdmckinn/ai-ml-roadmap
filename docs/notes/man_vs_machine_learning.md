---
title: 'Man vs Machine Learning'
date: '2024-12-28'
author: 'Tirell Mckinnon'
description: 'Early reflections on machine learning from a software engineer's perspective'
tags: ['machine-learning', 'software-development', 'learning']
---

# Man vs Machine Learning

## The Race to the Bottom
I started thinking about AI models and noticed something interesting and agreed with a term called "race to the bottom". Potentially once all these models have access to the same information, they'll probably give pretty similar outputs. It's like when you search YouTube for how to code something - different teachers explain it differently, but the core information is the same or even cooking recipes. The only real difference is how it's delivered and the populary itself is even dependent on how well the information is reciveved. Theres also variance that can be built into these models with temperature that allows for randmoness and creativness for a while I think humans still when here but it is amazing that this can be built into models.

But what does "race to the bottom" really mean here? Sure, some companies will always have advantages with their proprietary data - that's their edge. But there's so much open source information out there - historical data, public statistics, market trends, the INTERNET - that most models will have access to similar foundational knowledge. They'll probably converge in many areas, with specific advantages in certain domains based on their unique data or training. T

This got me thinking about learning in this context. If models are going to be similar in their base capabilities, maybe the real value isn't just in having information, but in knowing how to use it, combine it, and build with it. It's like how knowing syntax isn't enough to be a good engineer - you need to understand systems, patterns, and how to solve real problems.

## Learning Types: Machines and Me
This got me wondering about the whole point of learning in an AI world. As a software engineer getting into AI/ML, I found myself making connections between how machines learn and how I learn. I'm a visual and hands-on learner - I need diagrams to understand systems and actual practice to make things click.

Looking at the basic types of machine learning, they map surprisingly well to different ways of learning:

![Learning Style Parallels](/diagrams/ml-parallels.svg)

Supervised learning is about starting with labeled data - you've got a base to build from with someone guiding you along the way. It reminds me of when I first picked up Go going through pratical examples, going through Jon Calhoun Go lang course, youtube channels, and guided working code. 

Unsupervised learning is more like being handed the docs and drawing your own conclusions, for machines this means unlabled data. This hits home for me when I'm trying to understand a new codebase. No one's telling me what patterns to look for - I'm digging through files, connecting the dots between services, and those "aha!" moments come when I start seeing how everything fits together. Usually by actually trying to run a working program console log or just generally break it so I can debug. 

## The Bootstrap Problem
Here's what I've figured out though - you need to know enough to even get started. I call this the bootstrap problem. Because I'm a software engineer, I can look at ML code and even if I don't know Python well, I get the structure enough to know what questions to ask. That baseline makes it possible to learn more.

![Bootstrap Effect](/diagrams/bootstrap-learning.svg)

This isn't just theory - I saw it in action recently when looking at some ML code for a plant compatibility analyzer. Even though I didn't know Python well, my software background helped me recognize patterns and ask meaningful questions. Things like "How do we test ML outputs?" and "How do we handle variability in the results?" These aren't just random questions - they come from years of software development experience.

![Pattern Recognition](/diagrams/code-recognition.svg)

Reinforcement learning is another machine learning type and it really drives this home for me. Like that recent hackathon where I was building a favorites system in Elixir. I'm not an Elixir dev, but this is where that bootstrap problem I mentioned really showed up - knowing other programming languages gave me enough foundation to start. Sure, I hit every possible error (that's my superpower), but because I understood basic programming concepts, I could use AI effectively to help me debug and learn. I wasn't just copying solutions - I was understanding them because I had the right context. Context is key in everything as it gives you a place to build from.

The learning curve was real - spent the first three days constantly looking up syntax, but by days four and five, I was pretty much self-reliant. I ended up with a working demo that was "almost production ready" 😅 (hackathon production ready or 2 sprints estimated), which isn't bad for an engineering manager who doesn't get to code as much these days. This whole experience really showed me how having a technical foundation lets you learn efficiently, even with AI help. You still need that baseline understanding to ask the right questions and make sense of the answers.

## Building vs Using
This circles back to what's actually valuable. Sure, AI models might all end up with the same knowledge, and you can find tutorials for pretty much anything. But there's a big difference between using tools and knowing how to build them. I'm a builder - I create tools. That means I need to understand how things work under the hood, not just how to use them.

I'm diving into ML types through Coursera now, and my mind's already racing with ideas for projects - like that plant compatibility analyzer. I'll probably have completely different insights next week, but that's the point, right? Keep building, keep learning, keep moving forward. As someone on X wisely said:

> "You can just learn things. No one is going to stop you."
