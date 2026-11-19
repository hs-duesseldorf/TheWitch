using UnrealBuildTool;
using System.Collections.Generic;

public class TheWitchTarget : TargetRules
{
	public TheWitchTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.V7;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		ExtraModuleNames.Add("TheWitch");
	}
}
