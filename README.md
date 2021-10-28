# Understanding Galaxies

MPhys project repository for generating redshifted images and making CNN predictions on galaxy morphology

Useful cluster commands

Local code to cluster

    rsync -azv -e 'ssh -A -J walml@external.jb.man.ac.uk' --exclude '*.png' understanding_galaxies walml@galahad.ast.man.ac.uk:/share/nas/walml/repos

Predictions to local

    rsync -azv -e 'ssh -A -J walml@external.jb.man.ac.uk' --exclude '*.png' walml@galahad.ast.man.ac.uk:/share/nas/walml/repos/understanding_galaxies/results understanding_galaxies
