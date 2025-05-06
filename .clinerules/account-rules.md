# Account feature

## Roles
- Elderly account: Limited functionality, simplified UI. Can bind to multiple General account.
- General account: Full functionality. Can access elderly data bind to its account.

- First login: Redirect to the page doing the onboarding process, which let the user choose its role, save the role settings to the user's settings file.
- General login: Show different content based on the user's role.

## Binding Flow

1. General account can bind another accounts(elderly or general). If role changed to elderly then delete all the bindings before.
2. General account user can add users to bind by their email.
3. The account being added to the invite list will receive a confirm message about the binding.
4. If accept, then the binding created.
