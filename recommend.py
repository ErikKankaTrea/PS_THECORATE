from utils import *
import argparse

def parse_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description='DistanceSimilarity')

    parser.add_argument(
        '--url',
        dest='image_to_measure',
        default='',
        required=True)
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    
    # Read data
	url = args.image_to_measure
	master_df=pd.read_csv(os.path.join(get_data_path(), 'master_table.csv'), sep=";")
	master_df['img_repr'] = master_df.iloc[:, 2:-1].values.tolist()
	# Init class given picture
	SimTool=SimDistImg(url, ds=master_df, top_sim=6)
	# Get imgs and similarities
	base_image, similar_img_ids, similar_images_df = SimTool.get_similar_images_annoy()
	images = [imgs_to_pil(os.path.join(get_data_path(), 'webs_props', i+'.jpg')) for i in similar_images_df.uuid]
	features_df = pd.read_csv(os.path.join(get_data_path(), 'features.csv'),sep=";")
	# Save
	SimTool.save_sim_images(images, data=similar_images_df, features_df=features_df)
	