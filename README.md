# gym-tictactoe

OpenAI Gym Style Tic-Tac-Toe Environment.

	  |O|X
	 -----
	  |O| 
	 -----
	 O|X|X

	O's turn.
	Enter location[1-9], q for quit:

## Requirement

Python >= 3.5

## Install

    git clone https://github.com/haje01/gym-tictactoe.git
    cd gym-tictactoe/
    pip install -e .


## Try example agents

    cd examples/
    python human_agent.py
    python base_agent.py
    python td_agent.py


# Temporal Difference Agent Commands

## Learn

	Usage: td_agent.py learn [OPTIONS]

	  Learn and save the model.

	Options:
	  -p, --episode INTEGER  Episode count.  [default: 17000]
	  -e, --epsilon FLOAT    Exploring factor.  [default: 0.08]
	  -a, --alpha FLOAT      Step size.  [default: 0.4]
	  -f, --save-file TEXT   Save model data as file name.  [default:
							 td_agent.dat]
	  --help                 Show this message and exit.

## Bench

	Usage: td_agent.py bench [OPTIONS]

	  Benchmark agent with base agent.

	Options:
	  -p, --episode INTEGER  Episode count.  [default: 3000]
	  -f, --model-file TEXT  Model data file name.  [default: td_agent.dat]
	  --help                 Show this message and exit

## Grid search

	Usage: td_agent.py gridsearch [OPTIONS]

	  Grid search hyper-parameters.

	Options:
	  -q, --quality [high|mid|low]  Grid search quality.  [default: mid]
	  -r, --reproduce-test INTEGER  Reproducibility test count.  [default: 3]
	  --help                        Show this message and exit.

## Play

	Usage: td_agent.py play [OPTIONS]

	  Play with human.

	Options:
	  -f, --load-file TEXT  Load file name.  [default: td_agent.dat]
	  -n, --show-number     Show location number when play.  [default: False]
	  --help                Show this message and exit.
