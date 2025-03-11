let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.pandas
      python-pkgs.requests
      python-pkgs.numpy
      python-pkgs.pytorch
      python-pkgs.pymongo
      python-pkgs.python-dotenv
      python-pkgs.openai

      python-pkgs.torch
      python-pkgs.transformers
      python-pkgs.datasets
      python-pkgs.sentencepiece
      python-pkgs.accelerate
      python-pkgs.evaluate
      python-pkgs.tensorboard
    ]))
  ];
  shellHook = ''
    export PATH="$PATH:$(pwd)/backend"
    export PYTHONPATH="$(pwd)/backend"    
  '';
}
