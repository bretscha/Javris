# Javris Prediction and Control algorithms
## Javris Control and environment run.

Before use:

Start the robot-scripts.
Change the init variables in the run.py script to train, evaluate the Javris Control, or play yourself.
Create a conda environment and import everything from the requirements.txt.
Alternatively use the already implemented environment under the attacker pc as follows.
```
user@user:~$ ssh poker
comnets@comnets-PC:~$ cd git/catchmeifyoucan_ai_gym/
comnets@comnets-PC:~$ conda activate catchmeifoucan
comnets@comnets-PC:~$ python run.py

```
(Yes the conda environment is called catchmeifoucan with no y... so vorsicht. )
And have fun watching the Javris Control.

## Javris Prediction
To train a new Javris Prediction algorithm use the already existing prediction.ipynb notbook to do so.
After training import the model into 
```
catchmeifyoucan_ai_gym/gym_game/ai/prediction
```
and change the model in the run.py line 117 to the desired prediction algorithm.
