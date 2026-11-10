"""Generate explicit screening configurations from the snapshotted v1 config."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write(name, config):
    (ROOT / name).write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')


def main():
    base = json.loads((ROOT / 'original_config.json').read_text())
    base.update(study_version=2, profile='v2_development_only', seeds=[11, 23],
                cpu_fraction=.7, gpu_ids=[0], paper_role='exploratory development screening')
    scaled = dict(name='scaled_aligned_ce', architecture='spatial', loss='ce',
                  spatial_bias=True, residual_gain_init=.001)
    warm = dict(scaled, name='scaled_warm_aligned_ce', warmup_epochs=3)
    controls = [dict(name='gated_ce', architecture='gated', loss='ce'),
                dict(name='aligned_ce', architecture='spatial', loss='ce', spatial_bias=True)]
    screen = copy.deepcopy(base)
    screen.update(epochs=10, recipes=controls + [scaled, warm])
    write('config_screen_ce.json', screen)
    for selected in controls + [scaled, warm]:
        config = copy.deepcopy(base)
        config.update(epochs=30, selected_architecture=selected['name'])
        config['recipes'] = []
        for kind in ['ce', 'blend_uniform', 'blend_tempered', 'pfl']:
            recipe = dict(selected, name=selected['name'] + '_' + kind, loss=kind)
            if kind.startswith('blend_'):
                recipe.update(blend_ramp_epochs=10, blend_max=.5, focal_gamma=1.)
            config['recipes'].append(recipe)
        write('config_loss_' + selected['name'] + '.json', config)
    print('Generated one CE screen and four alternative loss screens. Select exactly one loss screen.')


if __name__ == '__main__':
    main()
