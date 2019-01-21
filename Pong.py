import turtle
import numpy as np
import random
import math
import matplotlib.pyplot as plt

MAXSTATES = 10**3
GAMMA = 0.9
ALPHA = 0.01

#Setting up the turtle screen
w=turtle.Screen()
w.setup(width=800,height=600)
w.bgcolor("black")
w.tracer(0)

#p_b is the paddle on the right
p_b=turtle.Turtle()
p_b.speed(0)
p_b.shape("square")
p_b.color("blue")
p_b.shapesize(stretch_wid=5,stretch_len=1)
p_b.penup()
p_b.goto(350,0)

#Creating the ball object
ball=turtle.Turtle()
ball.speed(0)
ball.shape("circle")
ball.color("yellow")
ball.penup()
ball.goto(0,0)
ball.dx=1
ball.dy=-1

#these 2 functions move the paddle up and down
def pb_up():
    y=p_b.ycor()
    y=y+2
    p_b.sety(y)
def pb_down():
    y=p_b.ycor()
    y=y-2
    p_b.sety(y)


def max_dict(d):
	max_v = float('-inf')
	for key, val in d.items():
		if val > max_v:
			max_v = val
			max_key = key
	return max_key, max_v

def create_bins():

	bins = np.zeros((2,10))
	bins[0] = np.linspace(-390, 380, 10)
	bins[1] = np.linspace(-280, 290, 10)

	return bins

def assign_bins(observation, bins):
	state = np.zeros(2)
	for i in range(2):
		state[i] = np.digitize(observation[i], bins[i])
	return state

def get_state_as_string(state):
	string_state = ''.join(str(int(e)) for e in state)
	return string_state

def get_all_states_as_string():
	states = []
	for i in range(MAXSTATES):
		states.append(str(i).zfill(2))
	return states

def initialize_Q():
	Q = {}

	all_states = get_all_states_as_string()
	for state in all_states:
		Q[state] = {}
		for action in range(2):
			Q[state][action] = 0
	return Q

def play_one_game(bins, Q, eps=0.5):
	observation = [ball.xcor(),ball.ycor()]
	done = False
	cnt = 0 # number of moves in an episode
	state = get_state_as_string(assign_bins(observation, bins))
	total_reward = 0

	while not done:
                cnt += 1

                w.update()

                observation[0] = ball.xcor()
                observation[1] = ball.ycor()

                if np.random.uniform() < eps:
                        act = random.randint(0,1) # epsilon greedy
                else:
                        act = max_dict(Q[state])[0]

                if act==0:
                        pb_down()
                else:
                        pb_up()

                reward=0
                if ball.ycor() > 290:
                    ball.sety(290)
                    ball.dy *= -1
                if ball.ycor() < -280:
                    ball.sety(-280)
                    ball.dy *= -1
                if ball.xcor() > 380:  # this is when we loose the game
                    ball.goto(0, 0)
                    ball.dx *= -1
                    ball.dy *= -1
                    done = True
                if ball.xcor() < -390:
                    ball.setx(-390)
                    ball.dx *= -1
                if ball.xcor() > 330 and ball.xcor() < 340 and ball.ycor() < p_b.ycor() + 40 and ball.ycor() > p_b.ycor() - 40:
                    ball.setx(330)
                    ball.dx *= -1
                    reward = 200
                if p_b.ycor() > 250:
                    p_b.sety(250)
                if p_b.ycor() < -250:
                    p_b.sety(-250)
                    
                ball.setx(ball.xcor()+ball.dx)
                ball.sety(ball.ycor()+ball.dy)

                if done:
                        reward = -100

                total_reward += reward

                state_new = get_state_as_string(assign_bins(observation, bins))
	
                a1, max_q_s1a1 = max_dict(Q[state_new])
                Q[state][act] += ALPHA*(reward + GAMMA*max_q_s1a1 - Q[state][act])
                state, act = state_new, a1					

	return total_reward+100, cnt

def play_many_games(bins, N=3000):

	Q = initialize_Q()

	length = []
	reward = []
	k=0
	for n in range(N):
		eps=0.5/(1+n*10e-3)
		episode_reward, episode_length = play_one_game(bins, Q, eps)
		print(n, '  %.4f  ' % eps, episode_reward)
		length.append(episode_length)
		reward.append(episode_reward)
		if episode_reward!=0:
			k+=1
		else:
			k=0
		if k==10:
			break

	return length, reward

def plot_running_avg(totalrewards):
	N = len(totalrewards)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
	plt.plot(running_avg)
	plt.title("Running Average")
	plt.show()

if __name__ == '__main__':
	bins = create_bins()
	episode_lengths, episode_rewards = play_many_games(bins)

	plot_running_avg(episode_rewards)
