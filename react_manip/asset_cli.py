"""Prepare the checksum-locked PiPER-X assets outside the installed package."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare',))
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    from react_manip.robots.agilex_robosuite import asset_dir, validate_bundle
    from scripts.build_agilex_assets import Builder
    Builder(asset_dir(), args.offline).build()
    validate_bundle()
    print(f'PiPER-X assets ready: {asset_dir()}')


if __name__ == '__main__':
    main()
