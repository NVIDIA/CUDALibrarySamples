# Contribution rules

- There is no strict coding standard for contributions to this repository for the code mixture of C, C++, Python and other languages for different libraries. However, the general rule is to keep sample code consistent with other samples for the same project, whenever this is possible. When a completely new type of samples is introduced, we would expect the code to follow one of the standard coding guidelines (e.g., for C++ it would be[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)). In case there are any questions about the code style, we are open to discussion.
- Avoid introducing unnecessary complexity into existing code so that maintainability and readability are preserved.
- Try to keep pull requests (PRs) as concise as possible:
  - Avoid committing commented-out code.
  - Wherever possible, each commit should address a single concern. If there are several otherwise-unrelated things that should be changed to reach a desired endpoint, it is perfectly fine to open several PRs and state in the description which PR depends on another PR. The more complex the changes are in a single PR, the more time it will take to review those changes.
- Write PR and commit titles using imperative mood.
- Make sure that the build log is clean, meaning no warnings or errors should be present.
- If the new sample code introduces new non-trivial dependencies (either on external components or on a specific version of the library) which are different from other samples for the same project or not mentioned in the documentation, the new dependencies and implied limitations should be clearly stated, preferably with checks to guard potential users from encountering an unexpected behavior. 
- The contributed code should be tested to compile and run successfully in a clean environment with all non-trivial dependencies explicitly listed in the corresponding build scripts. If the contributed code is a new sample, the limitations on when the code is supposed to work must be listed explicitly (e.g., what types of inputs are expected)
- Make sure that you can contribute your work to open source (no license and/or patent conflict is introduced by your code). The code in this repository is licensed under <link to LICENSE>.
- You need to [`sign`](#Sign) your commit.
- Thanks in advance for your patience as we review your contributions; we do appreciate them!

<a name="Sign"></a>Sign your Work
--------------


We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be accepted.

To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

    $ git commit -s -m "Add cool feature."

This will append the following to your commit message:

    Signed-off-by: Your Name <your@email.com>

By doing this you certify the below (the original at https://developercertificate.org/):

    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
