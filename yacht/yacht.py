import env.yacht as Yacht

env = Yacht.YachtEnv()

state = env.reset()

print(env.action_space)
print(env.observation_space)

while True:
    env.render()
    
    action_reroll = input('Reroll(Y/N): ')
    if action_reroll.lower() == 'n':
        action_choose_number = int(input('Choose Number(1-12): '))
        action = [action_reroll, action_choose_number]
    elif action_reroll.lower() == 'y':
        action_reroll_dice = (input('Select Reroll Dice(1-5) (EX:1 3 5): ')).split()
        action = [action_reroll, action_reroll_dice]
    else:
        continue
    state, reward, done, info = env.step(action)
    if done:
        break