import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

def main():
    tflib.init_tf()

    #model_path = "./models/trypophobia.pkl"#密集恐惧症
    #model_path = "./models/generator_model-stylegan2-config-f.pkl"#模特
    #model_path = "./models/2020-01-11-skylion-stylegan2-animeportraits-networ.pkl"
    #model_path = "./models/2019-03-08-stylegan-animefaces-network-02051-021980.pkl"
    #model_path = "./models/generator_wanghong-stylegan2-config-f.pkl"
    model_path = "./models/generator_asian_star.pkl"
    #model_path = "./models/reimu.pkl"
    with open(model_path,"rb") as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)


    
    print("类型为")
    print(Gs.input_shape[1])
    print(type(Gs.input_shape[1]))
    rnd = np.random.RandomState(788888)
    latent_vector1 = rnd.randn(1, Gs.input_shape[1])
    
    number_of_frames =30

    frame_step = 1/number_of_frames
    iname = 0
    for all_count in range(1,120):
        x = 0
        rnd = np.random.RandomState(all_count*88+8788)
        latent_vector2 = rnd.randn(1, Gs.input_shape[1])#rnd.randn(1, Gs.input_shape[1])
        for frame_count in range(1,number_of_frames):
            x = x + frame_step
            latent_input = latent_vector1.copy()
            for i in range(512):
                f1 = latent_vector1[0][i]
                f2 = latent_vector2[0][i]
                #if f1 > f2:
                #    tmp = f2
                #    f2 = f1
                #    f1 = tmp
                fnew = f1 + (f2-f1)*x
                latent_input[0][i] = fnew
            inputimg = latent_input.copy()
            images = Gs.run(latent_input, None, truncation_psi=1, randomize_noise=False, output_transform=fmt)   
        # Save image.
            iname = iname+1
            os.makedirs(config.result_dir, exist_ok=True)
            png_filename = os.path.join(config.result_dir, 'animation_'+str(iname)+'.png')
            PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        latent_vector1 = inputimg.copy()
if __name__ == "__main__":
    main()