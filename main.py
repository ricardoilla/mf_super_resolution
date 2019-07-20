__author__ = 'RicardoIlla'

from classes import *

dirNames = ['data/aligned', 'data/denoised','data/filtered', 'data/upsampled', 'data/results', 'input']

for dir in dirNames:
    # Create target directory & all intermediate directories if don't exists
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Directory ", dir,  " Created ")
    else:
        print("Directory ", dir,  " already exists")


print('############################################################')
print('Please put your data (multiple frames) into "input" directory.\nUse .tif extension and 24 bits depth images.\n'
      'To run again: Delete the complete "data" folder and place another dataset into "input" folder.')
print('############################################################')


step1 = Filter('input/*.tif', 'data/filtered/')
step2 = NLMDenoise('data/filtered/*.tif', 'data/denoised/')
step3 = Upsample('data/denoised/*.tif', 'data/upsampled/')
step4 = Align('data/upsampled/*.tif', 'data/aligned/')
step5 = Merge('data/aligned/*.tif', 'data/results/mean_fusion.tif')
step6 = Deblur('data/results', 'super_resolution.tif')

step_list = ['Complete Run', 'Only Filter', 'Only NLM Denoise', 'Only Upsampling', 'Only Alignment', 'Only Merge', 'Only Deblur', 'Run without Filtering']

while True:
    for x in range(len(step_list)):
        print('{} - {}'.format(x + 1, step_list[x]))
    try:
        var = int(input("\nEnter the selected step to execute (or 0 to exit)=>"))
        if var == 0:
            print("Finishing...")
            # text_file.close()
            break
        elif var == 1:
            step1.run()
            step2.run()
            step3.run()
            step4.run()
            step5.run()
            step6.run()
        elif var == 2:
            step1.run()
        elif var == 3:
            step2.run()
        elif var == 4:
            step3.run()
        elif var == 5:
            step4.run()
        elif var == 6:
            step5.run()
        elif var == 7:
            step6.run()
        elif var == 8:
            step2 = NLMDenoise('input/*.tif', 'data/denoised/')
            step3 = Upsample('data/denoised/*.tif', 'data/upsampled/')
            step4 = Align('data/upsampled/*.tif', 'data/aligned/')
            step5 = Merge('data/aligned/*.tif', 'data/results/mean_fusion.tif')
            step6 = Deblur('data/results', 'super_resolution.tif')
            step2.run()
            step3.run()
            step4.run()
            step5.run()
            step6.run()
        else:
            print('Invalid Value')
    except IndexError:
        print('Index Error')